from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = Path("../../logs/BraDD_AnySat_LinearProbing_SemSeg/csv/version_0/metrics.csv")
    out_dir = Path("./outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "bradd_loss_iou_curve.png"

    df = pd.read_csv(csv_path)

    # 取 epoch 级别训练 loss
    train_df = df[["epoch", "train/loss_epoch"]].dropna()
    train_df = train_df.drop_duplicates(subset=["epoch"], keep="last")

    # 取验证 loss 和验证 IoU
    val_df = df[["epoch", "val/loss", "val/IoU"]].dropna()
    val_df = val_df.drop_duplicates(subset=["epoch"], keep="last")

    if train_df.empty:
        raise ValueError("没有找到 train/loss_epoch，请检查 metrics.csv。")

    if val_df.empty:
        raise ValueError("没有找到 val/loss 或 val/IoU，请检查 metrics.csv。")

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=600)

    # 损失曲线统一使用同一种颜色
    loss_color = "#1f77b4"

    # 左轴：loss
    ax1.plot(
        train_df["epoch"],
        train_df["train/loss_epoch"],
        color=loss_color,
        linestyle="-",
        marker="o",
        linewidth=2,
        label="Train Loss",
    )

    ax1.plot(
        val_df["epoch"],
        val_df["val/loss"],
        color=loss_color,
        linestyle="--",
        marker="s",
        linewidth=2,
        label="Val Loss",
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 右轴：IoU
    ax2 = ax1.twinx()
    ax2.plot(
        val_df["epoch"],
        val_df["val/IoU"],
        color="#d62728",
        linestyle="-",
        marker="^",
        linewidth=2,
        label="Val IoU",
    )
    ax2.set_ylabel("Val IoU")

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title("BraDD AnySat Linear Probing: Loss and IoU Curves")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Figure saved to: {out_path}")


if __name__ == "__main__":
    main()