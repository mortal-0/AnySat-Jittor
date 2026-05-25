from __future__ import annotations

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Any

# 避免服务器无显示环境下 matplotlib / Qt 报错
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 避免 AutoDL 环境里 OMP_NUM_THREADS=0 导致 Jittor / OpenMP 警告
if os.environ.get("OMP_NUM_THREADS", "").strip() in ["", "0"]:
    os.environ["OMP_NUM_THREADS"] = "8"


def find_project_root() -> Path:
    """
    当前脚本建议放在:
        AnySat-main/src/visualization/evaluate_ts2c_confusion_matrices.py

    因此:
        parents[0] = visualization
        parents[1] = src
        parents[2] = AnySat-main
    """
    return Path(__file__).resolve().parents[2]


ROOT = find_project_root()

# 原项目配置中会使用 ${oc.env:PROJECT_ROOT}
os.environ["PROJECT_ROOT"] = str(ROOT)

# 同时加入项目根目录和 src 目录：
# 1. ROOT 用于支持 import src.models.module
# 2. SRC_DIR 用于支持 import data.datamodule 等原项目无 src 前缀的导入
SRC_DIR = ROOT / "src"

for p in [str(ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =========================
# 通用工具
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_methods(methods: str) -> list[str]:
    return [m.strip() for m in methods.split(",") if m.strip()]


def parse_class_names(class_names: str, num_classes: int) -> list[str]:
    if class_names.strip():
        names = [x.strip() for x in class_names.split(",")]
        if len(names) != num_classes:
            raise ValueError(
                f"class_names 数量为 {len(names)}，但 num_classes={num_classes}，二者不一致。"
            )
        return names

    return [str(i) for i in range(num_classes)]


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    y_true = y_true.reshape(-1).astype(np.int64)
    y_pred = y_pred.reshape(-1).astype(np.int64)

    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    return cm


def compute_metrics_from_cm(cm: np.ndarray) -> dict:
    total = cm.sum()
    acc = np.trace(cm) / max(total, 1)

    f1_list = []
    precision_list = []
    recall_list = []

    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else np.nan
        )

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "accuracy": float(acc),
        "macro_precision": float(np.nanmean(precision_list)),
        "macro_recall": float(np.nanmean(recall_list)),
        "macro_f1": float(np.nanmean(f1_list)),
        "num_samples": int(total),
    }


def save_cm_csv(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.index.name = "True\\Pred"
    df.to_csv(out_path, encoding="utf-8-sig")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
    normalize: bool = True,
) -> None:
    """
    绘制混淆矩阵。

    normalize=True 时按行归一化，即每一行表示某个真实类别被预测到各类别的比例。
    使用 Blues 色图，浅底黑字、深底白字，便于看清格子里的数字。
    """
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_show = cm / np.maximum(row_sum, 1)
        fmt = ".2f"
        colorbar_label = "Row-normalized ratio"
        vmin, vmax = 0.0, 1.0
    else:
        cm_show = cm.astype(float)
        fmt = "d"
        colorbar_label = "Count"
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(
        cm_show,
        interpolation="nearest",
        aspect="auto",
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # 添加网格线，让格子边界更明显
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if len(class_names) <= 25:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm_show[i, j]

                if normalize:
                    text = format(value, fmt)
                    # 归一化图中，深蓝区域用白字，其余用黑字
                    text_color = "white" if value >= 0.50 else "black"
                else:
                    text = str(int(cm[i, j]))
                    max_count = cm.max() if cm.max() > 0 else 1
                    text_color = "white" if cm[i, j] >= 0.50 * max_count else "black"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    df = pd.DataFrame(
        {
            "y_true": y_true.reshape(-1).astype(int),
            "y_pred": y_pred.reshape(-1).astype(int),
            "correct": (y_true.reshape(-1) == y_pred.reshape(-1)).astype(int),
        }
    )
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_summary(summary_rows: list[dict], out_path: Path) -> None:
    if not summary_rows:
        return

    keys = list(summary_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def save_method_results(
    method_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: list[str],
    out_dir: Path,
) -> dict:
    method_dir = out_dir / method_name
    ensure_dir(method_dir)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    metrics = compute_metrics_from_cm(cm)

    save_cm_csv(
        cm=cm,
        class_names=class_names,
        out_path=method_dir / "confusion_matrix_counts.csv",
    )

    save_predictions(
        y_true=y_true,
        y_pred=y_pred,
        out_path=method_dir / "test_predictions.csv",
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        out_path=method_dir / "confusion_matrix_normalized.png",
        title=f"{method_name} - Normalized Confusion Matrix",
        normalize=True,
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        out_path=method_dir / "confusion_matrix_counts.png",
        title=f"{method_name} - Confusion Matrix Counts",
        normalize=False,
    )

    row = {
        "method": method_name,
        **metrics,
    }

    print(f"\n[{method_name}]")
    print(f"accuracy      = {metrics['accuracy']:.6f}")
    print(f"macro_f1      = {metrics['macro_f1']:.6f}")
    print(f"macro_recall  = {metrics['macro_recall']:.6f}")
    print(f"macro_prec    = {metrics['macro_precision']:.6f}")
    print(f"num_samples   = {metrics['num_samples']}")

    return row


# =========================
# 1. 原项目 PyTorch E2E LP
# =========================

def find_checkpoint(ckpt_path: str, ckpt_dir: str) -> Path:
    if ckpt_path:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"指定的 checkpoint 不存在: {path}")
        return path

    ckpt_dir_path = Path(ckpt_dir)
    if not ckpt_dir_path.exists():
        raise FileNotFoundError(
            f"没有指定 --e2e_ckpt，且默认 checkpoint 目录不存在: {ckpt_dir_path}"
        )

    ckpts = list(ckpt_dir_path.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"目录下没有找到 .ckpt 文件: {ckpt_dir_path}")

    # 默认选修改时间最新的 ckpt。若你有明确 best ckpt，建议用 --e2e_ckpt 指定。
    ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


def move_to_device(x: Any, device):
    import torch

    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


def build_hydra_cfg(exp_name: str, data_dir: str, batch_size: int):
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass

    config_dir = str(ROOT / "configs")

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                f"exp={exp_name}",
                f"paths.data_dir={data_dir}",
                "logger=csv",
                "extras.enforce_tags=False",
                "test=False",
            ],
        )

    cfg.dataset.global_batch_size = batch_size
    return cfg


def eval_pytorch_e2e(
    exp_name: str,
    data_dir: str,
    ckpt_path: str,
    ckpt_dir: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import hydra

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    cfg = build_hydra_cfg(exp_name=exp_name, data_dir=data_dir, batch_size=batch_size)

    print("[E2E] Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    print("[E2E] Instantiating model...")
    model = hydra.utils.instantiate(cfg.model.instance)

    chosen_ckpt = find_checkpoint(ckpt_path=ckpt_path, ckpt_dir=ckpt_dir)
    print(f"[E2E] Loading checkpoint: {chosen_ckpt}")

    ckpt = torch.load(chosen_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device_obj)

    print("[E2E] Setting up test dataloader...")
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = move_to_device(batch, device_obj)

            logits = model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]

            if logits.ndim != 2:
                raise RuntimeError(
                    f"E2E 分类任务期望 logits 为 [B, C]，但得到 shape={tuple(logits.shape)}"
                )

            pred = logits.argmax(dim=1).detach().cpu().numpy()

            label = batch["label"]
            label = label.detach().cpu().numpy().reshape(-1)

            y_true_list.append(label)
            y_pred_list.append(pred)

            if batch_idx % 100 == 0:
                print(f"[E2E] processed batch {batch_idx}")

    y_true = np.concatenate(y_true_list, axis=0).astype(np.int64)
    y_pred = np.concatenate(y_pred_list, axis=0).astype(np.int64)

    return y_true, y_pred


# =========================
# 2. PyTorch Feature LP
# =========================

def eval_pytorch_feature(
    test_npz: str,
    ckpt_path: str,
    batch_size: int,
    num_classes: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class TS2CFeatureDataset(Dataset):
        def __init__(self, npz_path: str):
            data = np.load(npz_path, allow_pickle=True)
            self.x = data["x"].astype("float32")
            self.y = data["y"].astype("int64")

            if self.y.ndim > 1:
                self.y = self.y.reshape(self.y.shape[0], -1)
                if self.y.shape[1] == 1:
                    self.y = self.y[:, 0]

            self.x = torch.from_numpy(self.x)
            self.y = torch.from_numpy(self.y).long()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    class FeatureLPHead(nn.Module):
        def __init__(self, in_dim: int, num_classes: int):
            super().__init__()
            self.head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, num_classes),
            )

        def forward(self, x):
            return self.head(x)

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    test_set = TS2CFeatureDataset(test_npz)
    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    in_dim = int(test_set.x.shape[1])
    model = FeatureLPHead(in_dim=in_dim, num_classes=num_classes).to(device_obj)

    print(f"[PyTorch Feature] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device_obj, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device_obj, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).detach().cpu().numpy()

            y_true_list.append(y.numpy().reshape(-1))
            y_pred_list.append(pred.reshape(-1))

            if batch_idx % 100 == 0:
                print(f"[PyTorch Feature] processed batch {batch_idx}")

    y_true = np.concatenate(y_true_list, axis=0).astype(np.int64)
    y_pred = np.concatenate(y_pred_list, axis=0).astype(np.int64)

    return y_true, y_pred


# =========================
# 3. Jittor Feature LP
# =========================

def jittor_param_digest(model) -> float:
    """
    计算 Jittor 模型参数的绝对值和，用于判断加载前后参数是否发生变化。
    """
    total = 0.0

    try:
        state = model.state_dict()
        for _, v in state.items():
            try:
                total += float(np.abs(v.numpy()).sum())
            except Exception:
                pass
    except Exception:
        pass

    return total


def load_jittor_checkpoint_strict(model, ckpt_path: str) -> None:
    """
    严格加载 Jittor checkpoint。

    原来的 model.load(ckpt_path) 太宽松，可能出现没有真正加载成功但不报错的情况。
    这里优先使用 jt.load() 读取 state_dict，并检查 key 是否和当前模型匹配。
    """
    import jittor as jt

    ckpt_path = str(ckpt_path)

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Jittor checkpoint not found: {ckpt_path}")

    before_digest = jittor_param_digest(model)

    print(f"[Jittor Feature] Loading checkpoint strictly: {ckpt_path}")

    obj = jt.load(ckpt_path)

    if not isinstance(obj, dict):
        print("[Jittor Feature] jt.load result is not dict, fallback to model.load().")
        model.load(ckpt_path)
        after_digest = jittor_param_digest(model)
        print(f"[Jittor Feature] param digest before={before_digest:.6f}, after={after_digest:.6f}")

        if abs(after_digest - before_digest) < 1e-8:
            raise RuntimeError(
                "Jittor model.load() did not change parameters. "
                "Checkpoint may not be loaded correctly."
            )
        return

    # 兼容可能的包装格式
    if "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    elif "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())

    common_keys = sorted(model_keys & ckpt_keys)
    missing_keys = sorted(model_keys - ckpt_keys)
    unexpected_keys = sorted(ckpt_keys - model_keys)

    print(f"[Jittor Feature] model keys: {sorted(model_keys)}")
    print(f"[Jittor Feature] ckpt  keys: {sorted(ckpt_keys)}")
    print(f"[Jittor Feature] common keys: {common_keys}")
    print(f"[Jittor Feature] missing keys: {missing_keys}")
    print(f"[Jittor Feature] unexpected keys: {unexpected_keys}")

    if len(common_keys) == 0:
        raise RuntimeError(
            "Jittor checkpoint has zero matched keys with current model. "
            "You may be using a wrong checkpoint, or the model definition is inconsistent."
        )

    model.load_state_dict(state_dict)

    after_digest = jittor_param_digest(model)
    print(f"[Jittor Feature] param digest before={before_digest:.6f}, after={after_digest:.6f}")

    if abs(after_digest - before_digest) < 1e-8:
        raise RuntimeError(
            "Jittor checkpoint loading finished but parameters did not change. "
            "This usually means the checkpoint was not actually loaded."
        )

def eval_jittor_feature(
    test_npz: str,
    ckpt_path: str,
    batch_size: int,
    num_classes: int,
    use_cuda: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jittor cached-feature LP 测试集推理。

    重要：这里不再使用 jittor.dataset.Dataset。
    原因是当前报错显示 test_set.y 原始标签正常，但经过 Jittor Dataset 迭代后，
    y_true 中出现负数，导致 np.bincount 报错。

    因此这里采用最稳的方式：
        1. np.load 直接读取 x/y；
        2. y 保持 numpy 原始标签；
        3. x 手动分 batch 转成 jt.array；
        4. model(x) 得到 pred；
        5. 拼接 pred，计算混淆矩阵。
    """
    import jittor as jt
    import jittor.nn as nn

    jt.flags.use_cuda = int(use_cuda)

    class LPHeadJT(nn.Module):
        def __init__(self, in_dim: int, num_classes: int):
            super().__init__()
            self.norm = nn.LayerNorm(in_dim)
            self.fc = nn.Linear(in_dim, num_classes)

        def execute(self, x):
            x = self.norm(x)
            x = self.fc(x)
            return x

    # 1. 直接读取 npz，不走 Jittor Dataset
    data = np.load(test_npz, allow_pickle=True)
    x_all = data["x"].astype("float32")
    y_all = data["y"].astype("int64")

    if y_all.ndim > 1:
        y_all = y_all.reshape(y_all.shape[0], -1)
        if y_all.shape[1] == 1:
            y_all = y_all[:, 0]

    y_all = y_all.reshape(-1).astype(np.int64)

    in_dim = int(x_all.shape[1])

    print("[Jittor Feature] Loaded test feature npz directly")
    print(f"                 x shape = {x_all.shape}")
    print(f"                 y shape = {y_all.shape}")
    print(f"                 y min/max = {y_all.min()} / {y_all.max()}")
    print(f"                 y unique = {np.unique(y_all)}")
    print(f"                 y count  = {np.bincount(y_all, minlength=num_classes)}")
    print(f"                 in_dim = {in_dim}")
    print(f"                 num_classes = {num_classes}")

    # 这里提前拦截，避免后面混淆矩阵被脏标签污染
    invalid_label_mask = (y_all < 0) | (y_all >= num_classes)
    if invalid_label_mask.any():
        bad_values = np.unique(y_all[invalid_label_mask])
        raise ValueError(
            f"test_npz contains invalid labels: {bad_values}. "
            f"Expected labels in [0, {num_classes - 1}]."
        )

    # 2. 构建模型并严格加载 checkpoint
    model = LPHeadJT(in_dim=in_dim, num_classes=num_classes)

    load_jittor_checkpoint_strict(model, ckpt_path)

    model.eval()

    # 3. 手动 batch 推理
    y_pred_list = []
    n = x_all.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        x_batch_np = x_all[start:end]
        x_batch = jt.array(x_batch_np)

        logits = model(x_batch)
        logits_np = logits.numpy()

        pred = logits_np.argmax(axis=1).astype(np.int64)
        y_pred_list.append(pred)

        if (start // batch_size) % 100 == 0:
            print(f"[Jittor Feature] processed samples {start}/{n}")

    y_true = y_all.astype(np.int64)
    y_pred = np.concatenate(y_pred_list, axis=0).astype(np.int64)

    if len(y_true) != len(y_pred):
        raise RuntimeError(
            f"Prediction length mismatch: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}"
        )

    # 4. 再次检查预测是否越界
    invalid_pred_mask = (y_pred < 0) | (y_pred >= num_classes)
    if invalid_pred_mask.any():
        bad_values = np.unique(y_pred[invalid_pred_mask])
        raise ValueError(
            f"Model produced invalid predictions: {bad_values}. "
            f"Expected predictions in [0, {num_classes - 1}]."
        )

    print("[Jittor Feature] true label distribution:")
    print(np.bincount(y_true, minlength=num_classes))

    print("[Jittor Feature] pred label distribution:")
    print(np.bincount(y_pred, minlength=num_classes))

    acc = float((y_true == y_pred).mean())
    print(f"[Jittor Feature] raw test accuracy = {acc:.6f}")

    return y_true, y_pred

# =========================
# 主函数
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        type=str,
        default="e2e,pytorch_feature,jittor_feature",
        help="要评估的方法，用逗号分隔。可选: e2e,pytorch_feature,jittor_feature",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "src" / "visualization" / "outputs" / "ts2c_test_confusion_matrices"),
    )

    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument(
        "--class_names",
        type=str,
        default="",
        help="类别名，逗号分隔。默认使用 0,1,...,15。",
    )

    # 原项目 E2E
    parser.add_argument("--exp", type=str, default="TS2C_AnySat_LP")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    parser.add_argument(
        "--e2e_ckpt",
        type=str,
        default="",
        help="原项目 PyTorch E2E LP checkpoint 路径。为空则从 --e2e_ckpt_dir 自动选择最新 .ckpt。",
    )
    parser.add_argument(
        "--e2e_ckpt_dir",
        type=str,
        default=str(
            ROOT
            / "logs"
            / "TimeSen2Crop_AnySat_LinearProbing_SemSeg"
            / "checkpoints"
        ),
    )

    # Feature 文件
    parser.add_argument(
        "--test_npz",
        type=str,
        default=str(
            ROOT
            / "src"
            / "jittor_lp"
            / "outputs"
            / "ts2c_features"
            / "test_features.npz"
        ),
    )

    # PyTorch Feature LP
    parser.add_argument(
        "--pytorch_feature_ckpt",
        type=str,
        default=str(
            ROOT
            / "src"
            / "jittor_lp"
            / "outputs"
            / "pytorch_ts2c_lp_full"
            / "best_pytorch_feature_lp_head.pt"
        ),
    )

    # Jittor Feature LP
    parser.add_argument(
        "--jittor_feature_ckpt",
        type=str,
        default=str(
            ROOT
            / "src"
            / "jittor_lp"
            / "outputs"
            / "jittor_ts2c_lp_full"
            / "best_jittor_lp_head.pkl"
        ),
    )

    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--jittor_use_cuda", type=int, default=1)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    methods = parse_methods(args.methods)
    class_names = parse_class_names(args.class_names, num_classes=args.num_classes)

    print("[INFO] Project root:", ROOT)
    print("[INFO] Methods:", methods)
    print("[INFO] Output dir:", out_dir)

    summary_rows = []

    if "e2e" in methods:
        y_true, y_pred = eval_pytorch_e2e(
            exp_name=args.exp,
            data_dir=args.data_dir,
            ckpt_path=args.e2e_ckpt,
            ckpt_dir=args.e2e_ckpt_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        row = save_method_results(
            method_name="pytorch_e2e_lp",
            y_true=y_true,
            y_pred=y_pred,
            num_classes=args.num_classes,
            class_names=class_names,
            out_dir=out_dir,
        )
        summary_rows.append(row)

    if "pytorch_feature" in methods:
        y_true, y_pred = eval_pytorch_feature(
            test_npz=args.test_npz,
            ckpt_path=args.pytorch_feature_ckpt,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            device=args.device,
        )
        row = save_method_results(
            method_name="pytorch_feature_lp",
            y_true=y_true,
            y_pred=y_pred,
            num_classes=args.num_classes,
            class_names=class_names,
            out_dir=out_dir,
        )
        summary_rows.append(row)

    if "jittor_feature" in methods:
        y_true, y_pred = eval_jittor_feature(
            test_npz=args.test_npz,
            ckpt_path=args.jittor_feature_ckpt,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            use_cuda=args.jittor_use_cuda,
        )
        row = save_method_results(
            method_name="jittor_feature_lp",
            y_true=y_true,
            y_pred=y_pred,
            num_classes=args.num_classes,
            class_names=class_names,
            out_dir=out_dir,
        )
        summary_rows.append(row)

    save_summary(summary_rows, out_dir / "test_metrics_summary.csv")

    print("\n[DONE] All evaluation finished.")
    print(f"Summary saved to: {out_dir / 'test_metrics_summary.csv'}")
    print(f"Confusion matrices saved under: {out_dir}")


if __name__ == "__main__":
    main()