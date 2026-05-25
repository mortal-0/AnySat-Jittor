from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

# 服务器/无图形界面环境下防止 Qt 报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def find_project_root() -> Path:
    """
    寻找项目根目录
    """
    return Path(__file__).resolve().parents[2]


ROOT = find_project_root()


def read_original_pytorch_csv(path: str | Path) -> pd.DataFrame:
    """
    读取原项目 PyTorch Lightning 训练出来的 metrics.csv。

    Lightning 的 metrics.csv 往往是稀疏记录：
    - train/loss_epoch 只在 epoch 结束时有值
    - val/OA、val/loss 只在验证结束时有值
    - 其他 step 行可能是 NaN
    """
    path = Path(path)
    df = pd.read_csv(path)

    if "epoch" not in df.columns:
        raise ValueError(f"{path} 中没有 epoch 列，无法绘制曲线。")

    records = []

    all_epochs = sorted(df["epoch"].dropna().unique())

    for ep in all_epochs:
        sub = df[df["epoch"] == ep]

        row = {"epoch": int(ep)}

        # train loss
        if "train/loss_epoch" in sub.columns:
            vals = sub["train/loss_epoch"].dropna()
            row["train_loss"] = vals.iloc[-1] if len(vals) > 0 else None
        elif "train/cross_entropy_loss_epoch" in sub.columns:
            vals = sub["train/cross_entropy_loss_epoch"].dropna()
            row["train_loss"] = vals.iloc[-1] if len(vals) > 0 else None
        elif "train/loss" in sub.columns:
            vals = sub["train/loss"].dropna()
            row["train_loss"] = vals.iloc[-1] if len(vals) > 0 else None
        else:
            row["train_loss"] = None

        # val loss
        if "val/loss" in sub.columns:
            vals = sub["val/loss"].dropna()
            row["val_loss"] = vals.iloc[-1] if len(vals) > 0 else None
        elif "val/cross_entropy_loss" in sub.columns:
            vals = sub["val/cross_entropy_loss"].dropna()
            row["val_loss"] = vals.iloc[-1] if len(vals) > 0 else None
        else:
            row["val_loss"] = None

        # val accuracy / OA
        if "val/OA" in sub.columns:
            vals = sub["val/OA"].dropna()
            row["val_acc"] = vals.iloc[-1] if len(vals) > 0 else None
        elif "val/acc" in sub.columns:
            vals = sub["val/acc"].dropna()
            row["val_acc"] = vals.iloc[-1] if len(vals) > 0 else None
        elif "val/Accuracy" in sub.columns:
            vals = sub["val/Accuracy"].dropna()
            row["val_acc"] = vals.iloc[-1] if len(vals) > 0 else None
        else:
            row["val_acc"] = None

        # train accuracy / OA，原项目 csv 里不一定有
        if "train/OA" in sub.columns:
            vals = sub["train/OA"].dropna()
            row["train_acc"] = vals.iloc[-1] if len(vals) > 0 else None
        elif "train/acc" in sub.columns:
            vals = sub["train/acc"].dropna()
            row["train_acc"] = vals.iloc[-1] if len(vals) > 0 else None
        elif "train/Accuracy" in sub.columns:
            vals = sub["train/Accuracy"].dropna()
            row["train_acc"] = vals.iloc[-1] if len(vals) > 0 else None
        else:
            row["train_acc"] = None

        records.append(row)

    return pd.DataFrame(records)


def read_cached_feature_csv(path: str | Path) -> pd.DataFrame:
    """
    读取 PyTorch cached-feature / Jittor cached-feature 训练出来的 metrics.csv。
    这两个脚本通常包含:
        epoch, train_loss, train_acc, val_loss, val_acc
    """
    path = Path(path)
    df = pd.read_csv(path)

    if "epoch" not in df.columns:
        raise ValueError(f"{path} 中没有 epoch 列，无法绘制曲线。")

    keep_cols = ["epoch"]

    for col in ["train_loss", "val_loss", "train_acc", "val_acc"]:
        if col in df.columns:
            keep_cols.append(col)

    out = df[keep_cols].copy()
    out["epoch"] = out["epoch"].astype(int)

    return out


def smooth_series(s: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return s
    return s.rolling(window=window, min_periods=1).mean()


def plot_loss_curves(
    datasets: dict[str, pd.DataFrame],
    out_path: Path,
    smooth: int = 1,
):
    """
    只绘制验证集 loss 曲线，不绘制训练集 loss。
    """
    plt.figure(figsize=(9, 6))

    for name, df in datasets.items():
        if "val_loss" in df.columns and df["val_loss"].notna().any():
            plt.plot(
                df["epoch"],
                smooth_series(df["val_loss"], smooth),
                linestyle="-",
                linewidth=2.2,
                label=f"{name} Val Loss",
            )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def plot_val_acc_curves(
    datasets: dict[str, pd.DataFrame],
    out_path: Path,
    smooth: int = 1,
):
    plt.figure(figsize=(9, 6))

    for name, df in datasets.items():
        if "val_acc" in df.columns and df["val_acc"].notna().any():
            plt.plot(
                df["epoch"],
                smooth_series(df["val_acc"], smooth),
                linewidth=2.2,
                label=f"{name} Val Acc / OA",
            )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy / OA")
    plt.title("Validation Accuracy Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def plot_train_acc_curves(
    datasets: dict[str, pd.DataFrame],
    out_path: Path,
    smooth: int = 1,
):
    plt.figure(figsize=(9, 6))

    has_any = False

    for name, df in datasets.items():
        if "train_acc" in df.columns and df["train_acc"].notna().any():
            has_any = True
            plt.plot(
                df["epoch"],
                smooth_series(df["train_acc"], smooth),
                linewidth=2.2,
                label=f"{name} Train Acc",
            )

    if not has_any:
        plt.text(
            0.5,
            0.5,
            "No train accuracy column found.",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy Comparison")
    plt.grid(True, alpha=0.3)
    if has_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def plot_combined_figure(
    datasets: dict[str, pd.DataFrame],
    out_path: Path,
    smooth: int = 1,
):
    """
    左图只绘制验证集 loss；
    右图绘制验证集 accuracy / OA。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：Validation Loss
    ax = axes[0]
    for name, df in datasets.items():
        if "val_loss" in df.columns and df["val_loss"].notna().any():
            ax.plot(
                df["epoch"],
                smooth_series(df["val_loss"], smooth),
                linestyle="-",
                linewidth=2.2,
                label=f"{name} Val Loss",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 右图：Validation Accuracy / OA
    ax = axes[1]
    for name, df in datasets.items():
        if "val_acc" in df.columns and df["val_acc"].notna().any():
            ax.plot(
                df["epoch"],
                smooth_series(df["val_acc"], smooth),
                linewidth=2.2,
                label=f"{name} Val Acc / OA",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy / OA")
    ax.set_title("Validation Accuracy Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


def print_summary(datasets: dict[str, pd.DataFrame]):
    print("\n========== Summary ==========")

    for name, df in datasets.items():
        print(f"\n[{name}]")
        print(f"epochs: {int(df['epoch'].min())} -> {int(df['epoch'].max())}")

        if "train_loss" in df.columns and df["train_loss"].notna().any():
            print(f"final train_loss: {df['train_loss'].dropna().iloc[-1]:.6f}")

        if "val_loss" in df.columns and df["val_loss"].notna().any():
            print(f"best  val_loss:   {df['val_loss'].dropna().min():.6f}")
            print(f"final val_loss:   {df['val_loss'].dropna().iloc[-1]:.6f}")

        if "train_acc" in df.columns and df["train_acc"].notna().any():
            print(f"final train_acc:  {df['train_acc'].dropna().iloc[-1]:.6f}")

        if "val_acc" in df.columns and df["val_acc"].notna().any():
            best_idx = df["val_acc"].idxmax()
            best_epoch = int(df.loc[best_idx, "epoch"])
            best_val = float(df.loc[best_idx, "val_acc"])
            final_val = float(df["val_acc"].dropna().iloc[-1])
            print(f"best  val_acc:    {best_val:.6f} @ epoch {best_epoch}")
            print(f"final val_acc:    {final_val:.6f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original_csv",
        type=str,
        default=str(
            ROOT
            / "logs"
            / "TimeSen2Crop_AnySat_LinearProbing_SemSeg"
            / "csv"
            / "version_0"
            / "metrics.csv"
        ),
        help="原项目 PyTorch 端到端 LP 的 metrics.csv",
    )
    parser.add_argument(
        "--pytorch_feature_csv",
        type=str,
        default=str(
            ROOT
            / "src"
            / "jittor_lp"
            / "outputs"
            / "pytorch_ts2c_lp_full"
            / "metrics.csv"
        ),
        help="PyTorch cached-feature LP 的 metrics.csv",
    )
    parser.add_argument(
        "--jittor_feature_csv",
        type=str,
        default=str(
            ROOT
            / "src"
            / "jittor_lp"
            / "outputs"
            / "jittor_ts2c_lp_full"
            / "metrics.csv"
        ),
        help="Jittor cached-feature LP 的 metrics.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(
            ROOT
            / "src"
            / "visualization"
            / "outputs"
            / "ts2c_curve_comparison"
        ),
        help="输出图片保存目录",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="平滑窗口。1 表示不平滑。",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}

    original_csv = Path(args.original_csv)
    pytorch_feature_csv = Path(args.pytorch_feature_csv)
    jittor_feature_csv = Path(args.jittor_feature_csv)

    print("[INFO] Project root:", ROOT)
    print("[INFO] original_csv:", original_csv)
    print("[INFO] pytorch_feature_csv:", pytorch_feature_csv)
    print("[INFO] jittor_feature_csv:", jittor_feature_csv)
    print("[INFO] out_dir:", out_dir)

    if original_csv.exists():
        datasets["PyTorch E2E LP"] = read_original_pytorch_csv(original_csv)
    else:
        print(f"[Warning] original_csv not found: {original_csv}")

    if pytorch_feature_csv.exists():
        datasets["PyTorch Feature LP"] = read_cached_feature_csv(pytorch_feature_csv)
    else:
        print(f"[Warning] pytorch_feature_csv not found: {pytorch_feature_csv}")

    if jittor_feature_csv.exists():
        datasets["Jittor Feature LP"] = read_cached_feature_csv(jittor_feature_csv)
    else:
        print(f"[Warning] jittor_feature_csv not found: {jittor_feature_csv}")

    if not datasets:
        raise FileNotFoundError("No valid metrics.csv found. 请检查三个 CSV 路径是否正确。")

    # 保存统一后的 csv，方便检查字段是否对齐
    for name, df in datasets.items():
        safe_name = name.lower().replace(" ", "_")
        df.to_csv(out_dir / f"{safe_name}_normalized.csv", index=False)

    plot_loss_curves(
        datasets=datasets,
        out_path=out_dir / "loss_curves.png",
        smooth=args.smooth,
    )

    plot_val_acc_curves(
        datasets=datasets,
        out_path=out_dir / "val_accuracy_curves.png",
        smooth=args.smooth,
    )

    plot_train_acc_curves(
        datasets=datasets,
        out_path=out_dir / "train_accuracy_curves.png",
        smooth=args.smooth,
    )

    plot_combined_figure(
        datasets=datasets,
        out_path=out_dir / "combined_loss_valacc.png",
        smooth=args.smooth,
    )

    print_summary(datasets)

    print("\nSaved figures to:")
    print(f"  {out_dir / 'loss_curves.png'}")
    print(f"  {out_dir / 'val_accuracy_curves.png'}")
    print(f"  {out_dir / 'train_accuracy_curves.png'}")
    print(f"  {out_dir / 'combined_loss_valacc.png'}")


if __name__ == "__main__":
    main()