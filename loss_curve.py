import argparse
import ast
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Parse train/eval logs and plot metrics.")
    parser.add_argument("run_name", help="e.g. T1_d1_pretrained_train_36004")
    args = parser.parse_args()

    run_name = args.run_name
    log_file = os.path.join("logs", f"{run_name}.out")

    train_epochs, train_loss = [], []
    val_epochs, val_loss = [], []
    val_cer, val_wer = [], []

    # 讀檔與解析
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                data = ast.literal_eval(line)
            except Exception:
                continue

            # 訓練過程（逐 step 的 loss）
            if "loss" in data and "eval_loss" not in data and "epoch" in data:
                train_epochs.append(float(data["epoch"]))
                train_loss.append(float(data["loss"]))

            # 驗證/評估（逐次 eval）
            if "eval_loss" in data and "epoch" in data:
                val_epochs.append(float(data["epoch"]))
                val_loss.append(float(data["eval_loss"]))
                val_cer.append(float(data["eval_cer"])) if "eval_cer" in data else val_cer.append(None)
                val_wer.append(float(data["eval_wer"])) if "eval_wer" in data else val_wer.append(None)

    # 排序 epoch
    if train_epochs:
        pairs = sorted(zip(train_epochs, train_loss))
        train_epochs, train_loss = map(list, zip(*pairs))
    if val_epochs:
        pairs = sorted(zip(val_epochs, val_loss, val_cer, val_wer))
        val_epochs, val_loss, val_cer, val_wer = map(list, zip(*pairs))

    # 機率轉百分比
    def to_percent_if_prob(x):
        if x is None:
            return None
        return x * 100.0 if 0.0 <= x <= 1.0 else x

    val_cer = [to_percent_if_prob(x) for x in val_cer]
    val_wer = [to_percent_if_prob(x) for x in val_wer]

    # 輸出資料夾
    os.makedirs("plots", exist_ok=True)
    base = os.path.join("plots", run_name)

    # 圖1：Train vs Val Loss
    plt.figure(figsize=(10, 5))
    if train_epochs:
        plt.plot(train_epochs, train_loss, label="Train Loss", marker=".", linewidth=1)
    if val_epochs:
        plt.plot(val_epochs, val_loss, label="Val Loss", marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base}_losses.png")

    # 圖2：Val CER（Track 1）與 Val WER（Track 2）
    plt.figure(figsize=(10, 5))
    plotted_any = False

    if val_epochs and any(v is not None for v in val_cer):
        cer_x = [e for e, v in zip(val_epochs, val_cer) if v is not None]
        cer_y = [v for v in val_cer if v is not None]
        plt.plot(cer_x, cer_y, label="Track 1 CER (%)", marker="o")
        plotted_any = True

    if val_epochs and any(v is not None for v in val_wer):
        wer_x = [e for e, v in zip(val_epochs, val_wer) if v is not None]
        wer_y = [v for v in val_wer if v is not None]
        plt.plot(wer_x, wer_y, label="Track 2 WER (%)", marker="x")
        plotted_any = True

    if plotted_any:
        plt.xlabel("Epoch")
        plt.ylabel("Error (%)")
        plt.title("Validation CER/WER over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{base}_val_metrics.png")
    else:
        print("⚠️ 沒有在 log 中找到 eval_cer 或 eval_wer，因此略過 val_metrics 圖。")

    print("✅ 已輸出圖檔：")
    print(f" - {base}_losses.png")
    print(f" - {base}_val_metrics.png（若有 CER/WER）")

if __name__ == "__main__":
    main()
