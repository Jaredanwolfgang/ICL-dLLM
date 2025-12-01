import torch
import torch.nn.functional as F
# 假设你的模型代码在 src/model.py，类名为 ConditionedDiffusionTransformer
from models import DiffusionTransformerModel
from tqdm import tqdm
# 为了方便运行，我这里假设你已经定义好了 ConditionedDiffusionTransformer 类
# (请确保使用的是我们上一轮讨论的 "Conditioned" + "Interleaved" 版本)

def train_and_evaluate_icl():
    # --- 1. 配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    P = 20          # 序列长度 (19 demo + 1 query)
    D = 10          # 特征维度
    TOTAL_STEPS = 3000  # 增加训练步数
    PRINT_EVERY = 200
    
    # --- 2. 初始化模型 ---
    # 注意：Conditioned 版本通常收敛比 Joint 版本快
    model = DiffusionTransformerModel(
        n_positions=P, 
        n_x_dims=D, 
        n_y_dims=1,
        n_embd=256, 
        n_layer=4, 
        n_head=4, 
        timesteps=100  # 100步扩散足够了，太多会增加训练难度
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"Start Training on {DEVICE} for {TOTAL_STEPS} steps...")
    model.train()

    # --- 3. 训练循环 ---
    with tqdm(range(TOTAL_STEPS)) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            # === 动态生成数据 (Infinite Data) ===
            # 每次迭代生成不同的线性函数 y = xW
            xs = torch.randn(BATCH_SIZE, P, D, device=DEVICE)
            ws = torch.randn(BATCH_SIZE, D, 1, device=DEVICE) 
            ys = torch.bmm(xs, ws) # (B, P, 1)

            # === 归一化 (至关重要) ===
            # 计算每个 batch item 的均值方差进行归一化
            y_mean = ys.mean(dim=1, keepdim=True)
            y_std = ys.std(dim=1, keepdim=True) + 1e-8
            ys_norm = (ys - y_mean) / y_std

            # === 计算 Loss ===
            # 这里使用 Conditioned Forward
            loss, _, _, _ = model.compute_loss(xs, ys_norm)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪，防止爆炸
            optimizer.step()

            # === 打印日志 ===
            if step % PRINT_EVERY == 0:
                pbar.set_postfix(loss=loss.item())

    # --- 4. ICL 测试 (Inference) ---
    print("\nTraining Done. Running Evaluation...")
    model.eval()
    
    # 生成测试数据
    test_xs = torch.randn(BATCH_SIZE, P, D, device=DEVICE)
    test_ws = torch.randn(BATCH_SIZE, D, 1, device=DEVICE)
    test_ys = torch.bmm(test_xs, test_ws)
    
    # 归一化测试数据 (用于作为 ICL 的 Demo 输入)
    test_y_mean = test_ys.mean(dim=1, keepdim=True)
    test_y_std = test_ys.std(dim=1, keepdim=True) + 1e-8
    test_ys_norm = (test_ys - test_y_mean) / test_y_std

    # 构造 Mask: 前 P-1 个已知，最后一个未知
    mask_icl = torch.ones(BATCH_SIZE, P, 1, dtype=torch.bool, device=DEVICE)
    mask_icl[:, -1, :] = False

    # 运行采样 (假设你已经把 p_sample_loop 放在了类里，或者作为独立函数)
    # 这里需要确保 model 有 sample 或 p_sample_loop 方法
    # 如果没有，请把之前定义的 p_sample_loop 加到类里面
    with torch.no_grad():
        # 注意：这里传入 test_ys_norm 作为 ICL 参考
        y_pred_norm = model.p_sample_loop( # 假设方法名叫 sample_icl 或 sample
            test_xs, 
            shape=(BATCH_SIZE, P, 1),
            y_gt_icl=test_ys_norm, 
            mask_icl=mask_icl
        )[0]

    # 反归一化
    y_pred = y_pred_norm * test_y_std + test_y_mean

    # === 5. 最终指标 ===
    # 查询点误差 (Last Point)
    mse_query = F.mse_loss(y_pred[:, -1, :], test_ys[:, -1, :])
    
    # 演示点误差 (Demo Points) - 应该是 0
    mse_demo = F.mse_loss(y_pred[:, :-1, :], test_ys[:, :-1, :])

    print(f"Final Result:")
    print(f"  Demo Points MSE (Should be ~0): {mse_demo.item():.6f}")
    print(f"  Query Point MSE (Target < 1.0): {mse_query.item():.4f}")

    # 对比：如果是 Zero-Shot (瞎猜)，MSE 应该是多少？
    # 瞎猜均值(0)的误差大约等于 y 的方差
    variance_ref = test_ys[:, -1, :].var()
    print(f"  Reference Variance (Zero-Shot Baseline): {variance_ref.item():.4f}")

if __name__ == "__main__":
    train_and_evaluate_icl()