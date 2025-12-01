import argparse
import os
import uuid
from random import randint
from typing import Sequence

import torch
import yaml
from tqdm import tqdm

import wandb

from curriculum import Curriculum
from eval import get_run_metrics
from models import build_model
from samplers import get_data_sampler
from schema import Config, load_config
from tasks import get_task_sampler

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def diffusion_train_step(model, xs, ys, optimizer):
    optimizer.zero_grad()
    loss, eps_hat, y_t, t = model.compute_loss(xs, ys)
    loss.backward()
    optimizer.step()
    # For logging: compute masked ratio from time steps (higher t = more noise)
    # Approximate: average noise level from t values
    avg_t = t.float().mean().item() / model.timesteps
    return loss.detach().item(), eps_hat.detach(), avg_t

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = args.model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()
        
        if args.model.family in ["diffusion_gpt2", "diffusion_qwen2"]:
            y_mean = ys.mean(dim=1, keepdim=True)
            y_std = ys.std(dim=1, keepdim=True) + 1e-8
            ys_norm = (ys - y_mean) / y_std
            loss, eps_hat, masked_ratio = diffusion_train_step(model, xs.cuda(), ys_norm.cuda(), optimizer)
            # For pointwise loss, we need to reconstruct y predictions from noise predictions
            # This is approximate - in practice you'd need the full sampling process
            # For now, use eps_hat as a proxy (or compute y0_pred from y_t and eps_hat)
            output = eps_hat.squeeze(-1)  # Temporary: use noise predictions as proxy
        else:
            loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizer, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if (
            args.wandb.enabled
            and i % args.wandb.log_every_steps == 0
            and not args.test_run
        ):
            if args.model.family in ["diffusion_gpt2", "diffusion_qwen2"]:
                 wandb.log(
                    {
                        "overall_loss": loss,
                        "excess_loss": loss / baseline_loss,
                        "pointwise/loss": dict(
                            zip(point_wise_tags, point_wise_loss.cpu().numpy())
                        ),
                        "n_points": curriculum.n_points,
                        "n_dims": curriculum.n_dims_truncated,
                        "masked_ratio": masked_ratio,
                    },
                    step=i,
                )
            else:
                wandb.log(
                    {
                        "overall_loss": loss,
                        "excess_loss": loss / baseline_loss,
                        "pointwise/loss": dict(
                            zip(point_wise_tags, point_wise_loss.cpu().numpy())
                        ),
                        "n_points": curriculum.n_points,
                        "n_dims": curriculum.n_dims_truncated,
                    },
                    step=i,
                )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    elif args.wandb.enabled:
        wandb_kwargs = {
            "dir": args.out_dir,
            "project": args.wandb.project,
            "config": args.to_dict(),
            "notes": args.wandb.notes,
            "name": args.wandb.name,
            "resume": True,
        }
        if args.wandb.entity:
            wandb_kwargs["entity"] = args.wandb.entity
        try:
            wandb.init(**wandb_kwargs)
        except wandb.errors.CommError as exc:
            msg = (
                "Failed to initialise Weights & Biases run. "
                "Please ensure that `wandb.project`/`wandb.entity` are reachable "
                "and that you are logged in (`wandb login`). "
                f"Original error: {exc}"
            )
            raise RuntimeError(msg) from exc
    else:
        print("W&B logging disabled; continuing without initialising wandb.")

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


def _parse_overrides(values: Sequence[str]) -> Sequence[str]:
    overrides = []
    for value in values:
        if value.strip():
            overrides.append(value.strip())
    return overrides


def parse_cli() -> Config:
    parser = argparse.ArgumentParser(
        description="Train in-context learning models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/conf/base.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help=(
            "Override configuration values using dot notation, "
            "e.g. -o training.learning_rate=1e-4"
        ),
    )
    parsed = parser.parse_args()
    overrides = _parse_overrides(parsed.override)
    config = load_config(parsed.config, overrides)
    return config


if __name__ == "__main__":
    args = parse_cli()
    assert args.model.family in ["gpt2", "lstm", "qwen2.5", "diffusion_gpt2", "diffusion_qwen2"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as yaml_file:
            yaml.dump(args.to_dict(), yaml_file, default_flow_style=False)

    main(args)
