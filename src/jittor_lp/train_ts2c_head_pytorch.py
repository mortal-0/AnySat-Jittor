from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class TrainConfig:
    train_npz: str
    val_npz: str
    test_npz: str
    out_dir: str
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    num_workers: int
    limit_train_samples: int
    num_classes: int
    amp: bool
    scheduler: str
    warmup_epochs: int


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
        limit_samples: int | None = None,
    ):
        super().__init__()

        data = np.load(npz_path, allow_pickle=True)
        x = data["x"].astype("float32")
        y = data["y"].astype("int64")

        if y.ndim > 1:
            y = y.reshape(y.shape[0], -1)
            if y.shape[1] == 1:
                y = y[:, 0]

        if limit_samples is not None and limit_samples > 0:
            x = x[:limit_samples]
            y = y[:limit_samples]

        assert len(x) == len(y), f"x/y 样本数不一致: {len(x)} vs {len(y)}"

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class FeatureLPHead(nn.Module):
    """
    尽量对齐原项目 Fine_tuning 中的 LP head:

        LayerNorm(D) + Linear(D, num_classes)

    这里不再包含 AnySat backbone，因为输入已经是 backbone 输出特征。
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 为了复现性。速度优先时可以关掉 deterministic。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def infer_num_classes(*datasets: TS2CFeatureDataset, explicit_num_classes: int = 0) -> int:
    if explicit_num_classes and explicit_num_classes > 0:
        return int(explicit_num_classes)

    max_label = -1
    for ds in datasets:
        if ds is None:
            continue
        max_label = max(max_label, int(ds.y.max().item()))

    return max_label + 1


def build_loader(
    dataset: TS2CFeatureDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
):
    """
    默认 scheduler=none，用于和 Jittor fixed-lr 公平对比。

    如果想更接近 Lightning 原项目里常见的 warmup+cosine，可以用:
      --scheduler cosine
    """

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "none":
        return None

    if scheduler_name != "cosine":
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    warmup_epochs = max(0, int(warmup_epochs))
    total_epochs = max(1, int(epochs))

    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        if total_epochs <= warmup_epochs:
            return 1.0

        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = False,
) -> dict:
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        pred = logits.argmax(dim=1)

        batch_size = y.shape[0]
        total += batch_size
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * batch_size

    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config: TrainConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint_for_eval(
    model: nn.Module,
    path: Path,
    device: torch.device,
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return ckpt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--val_npz", type=str, required=True)
    parser.add_argument("--test_npz", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="outputs/pytorch_ts2c_feature_lp")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--limit_train_samples",
        type=int,
        default=0,
        help="调试用。0 表示使用全部训练样本。",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="默认自动从 train/val/test 标签推断。TimeSen2Crop 通常是 16。",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="是否使用 AMP。为了和 Jittor 头训练公平对比，默认关闭。",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="默认 none，用于和 Jittor fixed-lr 公平对比。",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    config = TrainConfig(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        test_npz=args.test_npz,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        limit_train_samples=args.limit_train_samples,
        num_classes=args.num_classes,
        amp=args.amp,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
    )

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4, ensure_ascii=False)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA 不可用，自动切换到 CPU。")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    limit_train = args.limit_train_samples if args.limit_train_samples > 0 else None

    train_set = TS2CFeatureDataset(args.train_npz, limit_samples=limit_train)
    val_set = TS2CFeatureDataset(args.val_npz)

    test_set = None
    if args.test_npz:
        test_set = TS2CFeatureDataset(args.test_npz)

    in_dim = int(train_set.x.shape[1])
    num_classes = infer_num_classes(
        train_set,
        val_set,
        test_set,
        explicit_num_classes=args.num_classes,
    )

    print("[INFO] Loaded feature datasets")
    print(f"       train x = {tuple(train_set.x.shape)}, y = {tuple(train_set.y.shape)}")
    print(f"       val   x = {tuple(val_set.x.shape)}, y = {tuple(val_set.y.shape)}")
    if test_set is not None:
        print(f"       test  x = {tuple(test_set.x.shape)}, y = {tuple(test_set.y.shape)}")
    print(f"       in_dim = {in_dim}")
    print(f"       num_classes = {num_classes}")
    print(f"       lr = {args.lr}, weight_decay = {args.weight_decay}")
    print(f"       scheduler = {args.scheduler}")
    print(f"       amp = {args.amp}")
    print(f"       device = {device}")

    train_loader = build_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = None
    if test_set is not None:
        test_loader = build_loader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = FeatureLPHead(in_dim=in_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = make_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_acc = -1.0
    best_epoch = -1

    metrics_rows = []
    best_model_path = out_dir / "best_pytorch_feature_lp_head.pt"

    for epoch in range(args.epochs):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = logits.argmax(dim=1)

            batch_size = y.shape[0]
            train_total += batch_size
            train_correct += int((pred == y).sum().item())
            train_loss_sum += float(loss.item()) * batch_size

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp=args.amp,
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]

        current_lr = optimizer.param_groups[0]["lr"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                config=config,
                path=best_model_path,
            )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
        }
        metrics_rows.append(row)
        save_metrics_csv(metrics_rows, out_dir / "metrics.csv")

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"[Epoch {epoch:03d}] "
                f"lr={current_lr:.8f}, "
                f"train_loss={train_loss:.6f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.6f}, val_acc={val_acc:.4f}, "
                f"best_val_acc={best_val_acc:.4f}@{best_epoch}"
            )

    print("[DONE] Training finished")
    print(f"       best_val_acc = {best_val_acc:.6f}")
    print(f"       best_epoch   = {best_epoch}")
    print(f"       best_model   = {best_model_path}")

    if test_loader is not None:
        print("[INFO] Loading best checkpoint for test...")
        ckpt = load_checkpoint_for_eval(model, best_model_path, device=device)
        best_epoch_loaded = ckpt.get("epoch", -1)

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            amp=args.amp,
        )

        print("[TEST] best checkpoint")
        print(f"       best_epoch = {best_epoch_loaded}")
        print(f"       test_loss  = {test_metrics['loss']:.6f}")
        print(f"       test_acc   = {test_metrics['acc']:.6f}")

        with open(out_dir / "test_result_best_model.txt", "w", encoding="utf-8") as f:
            f.write(f"best_epoch={best_epoch_loaded}\n")
            f.write(f"test_loss={test_metrics['loss']:.8f}\n")
            f.write(f"test_acc={test_metrics['acc']:.8f}\n")


if __name__ == "__main__":
    main()