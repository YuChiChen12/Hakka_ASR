import ast
import matplotlib.pyplot as plt
import os

# 替換成你的 .out 檔案路徑
log_file = "logs/2nd_finetune_35464.out"

# 初始化 lists
epochs = []
eval_loss = []
eval_cer = []

# 逐行處理
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = ast.literal_eval(line)  # 安全地轉成 dict
                if 'eval_loss' in data and 'eval_cer' in data and 'epoch' in data:
                    epochs.append(float(data['epoch']))
                    eval_loss.append(float(data['eval_loss']))
                    eval_cer.append(float(data['eval_cer']))
            except Exception as e:
                print(f"跳過錯誤行: {line}\n錯誤原因: {e}")

# 畫圖
plt.figure(figsize=(10, 5))
plt.plot(epochs, eval_loss, label="Eval Loss", marker="o")
plt.plot(epochs, eval_cer, label="Eval CER", marker="x")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Evaluation Loss and CER over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 儲存圖像
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/eval_plot.png")
print("✅ 圖像已儲存：plots/eval_plot.png")
