from __future__ import annotations

import os
import csv
import argparse
from pathlib import Path

import numpy as np
import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset


class TS2CFeatureDataset(Dataset):
    """
    读取由 PyTorch AnySat frozen backbone 导出的 .npz 特征。

    npz 内部要求：
      x: [N, D] float32
      y: [N] int64
    """

    def __init__(
        self,
        npz_path: str,
        batch_size: int = 128,
        shuffle: bool = False,
        limit_samples: int | None = None,
    ):
        super().__init__()

        data = np.load(npz_path, allow_pickle=True)
        self.x = data["x"].astype("float32")
        self.y = data["y"].astype("int64")

        if self.y.ndim > 1:
            self.y = self.y.reshape(self.y.shape[0], -1)
            if self.y.shape[1] == 1:
                self.y = self.y[:, 0]

        if limit_samples is not None and limit_samples > 0:
            self.x = self.x[:limit_samples]
            self.y = self.y[:limit_samples]

        assert len(self.x) == len(self.y), "x 和 y 样本数不一致"

        self.set_attrs(
            total_len=len(self.y),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class LPHead(nn.Module):
    """
    对齐 PyTorch 原项目中的 LP head：
        LayerNorm(D) + Linear(D, num_classes)
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, num_classes)

    def execute(self, x):
        x = self.norm(x)
        x = self.fc(x)
        return x


def accuracy_from_logits(logits: jt.Var, y: jt.Var) -> float:
    logits_np = logits.numpy()
    y_np = y.numpy().astype("int64")
    pred_np = logits_np.argmax(axis=1)
    return float((pred_np == y_np).mean())


def evaluate(model: nn.Module, loader: Dataset) -> dict:
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        logits = model(x)
        loss = nn.cross_entropy_loss(logits, y)

        logits_np = logits.numpy()
        y_np = y.numpy().astype("int64")
        pred_np = logits_np.argmax(axis=1)

        correct += int((pred_np == y_np).sum())
        total += int(len(y_np))
        loss_sum += float(loss.numpy()) * len(y_np)

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)

    return {
        "loss": avg_loss,
        "acc": acc,
    }


def save_metrics_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Jittor Module 通常支持 save；这里做一个兼容写法
    try:
        model.save(str(path))
    except Exception:
        jt.save(model.state_dict(), str(path))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_npz",
        type=str,
        required=True,
        help="train_features.npz 路径",
    )
    parser.add_argument(
        "--val_npz",
        type=str,
        required=True,
        help="val_features.npz 路径",
    )
    parser.add_argument(
        "--test_npz",
        type=str,
        default="",
        help="test_features.npz 路径，可选",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/jittor_ts2c_lp",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--use_cuda",
        type=int,
        default=1,
        help="1 表示使用 GPU，0 表示 CPU",
    )
    parser.add_argument(
        "--limit_train_samples",
        type=int,
        default=0,
        help="调试用：只取前 N 个训练样本；0 表示使用全部",
    )

    args = parser.parse_args()

    jt.flags.use_cuda = int(args.use_cuda)
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    limit_train = args.limit_train_samples if args.limit_train_samples > 0 else None

    train_set = TS2CFeatureDataset(
        args.train_npz,
        batch_size=args.batch_size,
        shuffle=True,
        limit_samples=limit_train,
    )
    val_set = TS2CFeatureDataset(
        args.val_npz,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_set = None
    if args.test_npz:
        test_set = TS2CFeatureDataset(
            args.test_npz,
            batch_size=args.batch_size,
            shuffle=False,
        )

    # 自动推断输入维度和类别数
    in_dim = int(train_set.x.shape[1])
    num_classes = int(np.max(train_set.y)) + 1

    print("[INFO] Loaded feature datasets")
    print(f"       train x = {train_set.x.shape}, y = {train_set.y.shape}")
    print(f"       val   x = {val_set.x.shape}, y = {val_set.y.shape}")
    if test_set is not None:
        print(f"       test  x = {test_set.x.shape}, y = {test_set.y.shape}")
    print(f"       in_dim = {in_dim}")
    print(f"       num_classes = {num_classes}")
    print(f"       lr = {args.lr}, weight_decay = {args.weight_decay}")

    model = LPHead(in_dim=in_dim, num_classes=num_classes)

    optimizer = jt.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    best_epoch = -1
    metrics_rows = []

    best_model_path = out_dir / "best_jittor_lp_head.pkl"

    for epoch in range(args.epochs):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_set:
            logits = model(x)
            loss = nn.cross_entropy_loss(logits, y)

            optimizer.step(loss)

            logits_np = logits.numpy()
            y_np = y.numpy().astype("int64")
            pred_np = logits_np.argmax(axis=1)

            train_correct += int((pred_np == y_np).sum())
            train_total += int(len(y_np))
            train_loss_sum += float(loss.numpy()) * len(y_np)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_metrics = evaluate(model, val_set)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": max(best_val_acc, val_acc),
        }
        metrics_rows.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_model(model, best_model_path)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.6f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}, "
                f"best_val_acc={best_val_acc:.4f}@{best_epoch}"
            )

        save_metrics_csv(metrics_rows, out_dir / "metrics.csv")

    print("[DONE] Training finished")
    print(f"       best_val_acc = {best_val_acc:.6f}")
    print(f"       best_epoch   = {best_epoch}")
    print(f"       best_model   = {best_model_path}")

    if test_set is not None:
        test_metrics = evaluate(model, test_set)
        print("[TEST] current final model")
        print(f"       test_loss = {test_metrics['loss']:.6f}")
        print(f"       test_acc  = {test_metrics['acc']:.6f}")

        with open(out_dir / "test_result_final_model.txt", "w", encoding="utf-8") as f:
            f.write(f"test_loss={test_metrics['loss']:.8f}\n")
            f.write(f"test_acc={test_metrics['acc']:.8f}\n")


if __name__ == "__main__":
    main()