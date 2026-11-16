import os
import subprocess
import time
import sys

# ================= 配置区域 =================
# 1. 设置使用的 GPU (如果有多个卡，可以在这里指定，例如 "0" 或 "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 2. 基础结果保存路径 (脚本会在这个目录下自动创建子文件夹)
BASE_SAVE_DIR = "./results_ablation_final"

# 3. 数据集名称
DATASET = "IEMOCAPFour"

# 4. 最佳超参数 (直接从你给的参数字典映射过来)
# 注意：不需要包含 --dataset, --ablation, --save_dir，这些会在循环中自动生成
BEST_PARAMS = {
    "--hidden_dim": "512",
    "--dropout": "0.45",
    "--p1_lr": "0.00022253",
    "--p1_wd": "0.001",
    "--p2_lr": "0.00049839",
    "--p3_gate_lr": "0.0016015",
    "--p3_gen_lr": "6.58e-05",
    "--p3_rec_weight": "0.06705",
    "--rdrop_alpha": "1.0",
    "--label_smoothing": "0.0",
    "--scheduler": "plateau",
    # 可以在这里添加 "--folds", "1" 如果只想跑 Fold 1 快速验证
}

# 5. 定义要运行的消融实验列表
ABLATION_MODES = [
    "none",             # 1. 完整模型 (Full Model / Baseline)
    "no_imputation",    # 2. 无补全 (w/o Imputation)
    "no_moe",           # 3. 无 MoE (w/o MoE)
    "no_gating",        # 4. 无门控 (w/o Gating)
    "no_hypergraph",    # 5. 无超图 (w/o Hypergraph)
    "no_causal_factor"  # 6. 无因果因子 (w/o Causal Factor)
]

# ================= 主程序 =================
def run_all_experiments():
    total = len(ABLATION_MODES)
    print(f"🚀 准备开始自动化实验，共 {total} 个任务。")
    print(f"📂 结果将保存在: {BASE_SAVE_DIR}")
    
    start_time = time.time()

    for idx, mode in enumerate(ABLATION_MODES):
        print(f"\n{'='*60}")
        print(f"▶️  [{idx+1}/{total}] 正在运行消融模式: {mode}")
        print(f"{'='*60}")

        # 1. 为每个消融实验创建独立的保存文件夹
        # 例如: ./results_ablation_final/no_moe
        current_save_dir = os.path.join(BASE_SAVE_DIR, mode)
        os.makedirs(current_save_dir, exist_ok=True)

        # 2. 构建命令行命令
        cmd = [
            sys.executable, "train_final_moe.py",  # 使用当前环境的 python
            "--dataset", DATASET,
            "--ablation", mode,
            "--save_dir", current_save_dir
        ]
        
        # 添加超参数
        for key, value in BEST_PARAMS.items():
            cmd.extend([key, value])

        # 打印即将执行的命令 (方便调试)
        print(f"执行命令: {' '.join(cmd)}")

        # 3. 执行命令
        # 我们使用 subprocess.run，这样可以实时看到输出，并且等待当前实验跑完再跑下一个
        try:
            # stdout=None 表示直接输出到终端，方便你监控进度
            # 如果你想静默运行并只保存到文件，可以改用 log 文件重定向，
            # 但你代码内部已经有了 get_logger 保存 train.log，所以这里直接看终端就行。
            subprocess.run(cmd, check=True)
            print(f"\n✅ 任务 [{mode}] 完成！日志已保存至 {current_save_dir}")
        
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 任务 [{mode}] 运行出错！(Exit Code: {e.returncode})")
            print("⚠️  跳过此任务，继续执行下一个...")
        
        except KeyboardInterrupt:
            print("\n🛑 用户手动中断。停止脚本。")
            break

        # 冷却时间，防止文件句柄未释放
        time.sleep(3)

    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"\n{'='*60}")
    print(f"🎉 所有实验结束！总耗时: {duration:.2f} 分钟")
    print(f"请查看 {BASE_SAVE_DIR} 目录下的各个子文件夹获取 train.log")

if __name__ == "__main__":
    run_all_experiments()