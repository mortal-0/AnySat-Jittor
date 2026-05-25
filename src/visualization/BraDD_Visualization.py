from __future__ import annotations

import os
import re
import sys
import csv
import argparse
from pathlib import Path
from typing import Any

# 清掉可能被 R / RStudio 污染的 Qt 环境变量
for k in ["QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_API"]:
    os.environ.pop(k, None)

# 强制 matplotlib 使用非 GUI 后端
import matplotlib
matplotlib.use("Agg")

import torch
import hydra
import pyrootutils
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# 定位项目根目录
ROOT = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# 把 src 目录显式加入 sys.path
SRC_DIR = Path(ROOT) / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OmegaConf.register_new_resolver("eval", eval)


def move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>| ]+', "_", str(name))


def norm_img(x: torch.Tensor) -> torch.Tensor:
    """
    用分位数拉伸代替简单 min-max，减少“看起来全黑”的假象。
    """
    x = x.float()
    flat = x.flatten()
    q1 = torch.quantile(flat, 0.01)
    q99 = torch.quantile(flat, 0.99)
    if (q99 - q1).abs() < 1e-8:
        return torch.zeros_like(x)
    x = x.clamp(q1, q99)
    return (x - q1) / (q99 - q1)


def get_valid_timesteps_from_mask(batch: dict, idx: int) -> list[int]:
    """
    优先根据 s1_mask 找有效时相。
    尽量兼容常见形状：
      [B, T]
      [B, T, C]
      [B, T, C, 1, 1]
    """
    if "s1_mask" not in batch:
        return []

    m = batch["s1_mask"][idx].detach().cpu()

    # 压到 [T, -1]
    if m.ndim == 1:
        valid = (m > 0).nonzero(as_tuple=False).flatten()
        return [int(v.item()) for v in valid]

    m = m.reshape(m.shape[0], -1)
    valid = (m.sum(dim=1) > 0).nonzero(as_tuple=False).flatten()
    return [int(v.item()) for v in valid]


def get_valid_timesteps_from_energy(s1: torch.Tensor) -> list[int]:
    """
    如果没有 s1_mask，就用非零能量近似判断有效时相。
    s1: [T, C, H, W]
    """
    energy = s1.abs().reshape(s1.shape[0], -1).sum(dim=1)
    valid = (energy > 0).nonzero(as_tuple=False).flatten()
    return [int(v.item()) for v in valid]


def choose_display_timestep(batch: dict, idx: int, mode: str = "last_valid") -> int:
    s1 = batch["s1"][idx].detach().cpu()  # [T, C, H, W]

    valid_ts = get_valid_timesteps_from_mask(batch, idx)
    if len(valid_ts) == 0:
        valid_ts = get_valid_timesteps_from_energy(s1)

    if len(valid_ts) == 0:
        return s1.shape[0] - 1

    if mode == "first_valid":
        return valid_ts[0]
    if mode == "middle_valid":
        return valid_ts[len(valid_ts) // 2]

    # 默认最后一个有效时相
    return valid_ts[-1]


def tensor_to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


def save_one_sample(
    batch: dict,
    pred: torch.Tensor,
    out_dir: Path,
    idx: int,
    timestep_mode: str = "last_valid",
    save_prob_map: bool = True,
) -> dict:
    """
    batch:
        - s1: [B, T, C, H, W]
        - label: [B, H, W]
        - name: list[str]
        - 可选: s1_mask
    pred:
        - [B, num_classes, H, W]
    """
    s1 = batch["s1"][idx].detach().cpu()               # [T, C, H, W]
    gt = batch["label"][idx].detach().cpu().long()     # [H, W]
    logits = pred[idx].detach().cpu()                  # [C, H, W]
    pred_mask = logits.argmax(dim=0).long()            # [H, W]

    # 前景概率图（假设二分类时前景类索引为 1）
    if logits.shape[0] >= 2:
        prob_fg = torch.softmax(logits, dim=0)[1]
    else:
        prob_fg = torch.sigmoid(logits[0])

    names = batch.get("name", None)
    if names is None:
        sample_name = f"sample_{idx}"
    else:
        sample_name = safe_name(names[idx])

    # 取有效时相，而不是盲目取最后一个时相
    t = choose_display_timestep(batch, idx, mode=timestep_mode)
    img_t = s1[t]  # [C, H, W]

    vv = norm_img(img_t[0])
    vh = norm_img(img_t[1]) if img_t.shape[0] > 1 else None
    ratio = norm_img(img_t[2]) if img_t.shape[0] > 2 else None

    gt_fg_ratio = tensor_to_float((gt > 0).float().mean())
    pred_fg_ratio = tensor_to_float((pred_mask > 0).float().mean())

    # 布局：VV / VH / Ratio / GT / Pred / Prob
    ncols = 6 if (ratio is not None and save_prob_map) else \
            5 if (ratio is None and save_prob_map) else \
            5 if (ratio is not None and not save_prob_map) else 4

    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3))

    ax_id = 0
    axes[ax_id].imshow(vv.numpy(), cmap="gray")
    axes[ax_id].set_title(f"VV (t={t})")
    axes[ax_id].axis("off")
    ax_id += 1

    if vh is not None:
        axes[ax_id].imshow(vh.numpy(), cmap="gray")
        axes[ax_id].set_title(f"VH (t={t})")
        axes[ax_id].axis("off")
        ax_id += 1

    if ratio is not None:
        axes[ax_id].imshow(ratio.numpy(), cmap="gray")
        axes[ax_id].set_title(f"Ratio (t={t})")
        axes[ax_id].axis("off")
        ax_id += 1

    axes[ax_id].imshow(gt.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[ax_id].set_title(f"GT mask\nfg={gt_fg_ratio:.4f}")
    axes[ax_id].axis("off")
    ax_id += 1

    axes[ax_id].imshow(pred_mask.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[ax_id].set_title(f"Pred mask\nfg={pred_fg_ratio:.4f}")
    axes[ax_id].axis("off")
    ax_id += 1

    if save_prob_map:
        axes[ax_id].imshow(prob_fg.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        axes[ax_id].set_title("Pred fg prob")
        axes[ax_id].axis("off")

    plt.tight_layout()
    save_path = out_dir / f"{sample_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "name": sample_name,
        "timestep": t,
        "gt_fg_ratio": gt_fg_ratio,
        "pred_fg_ratio": pred_fg_ratio,
    }


def build_cfg(exp_name: str, data_dir: str) -> DictConfig:
    config_dir = str(Path(ROOT) / "configs")

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                f"exp={exp_name}",
                f"paths.data_dir={data_dir}",
                "logger=csv",
                "test=False",
            ],
        )
    return cfg


def save_summary_csv(rows: list[dict], out_dir: Path) -> None:
    csv_path = out_dir / "summary.csv"
    if not rows:
        return

    keys = ["name", "timestep", "gt_fg_ratio", "pred_fg_ratio"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="训练得到的 checkpoint 路径，例如 logs/BraDD_AnySat_LinearProbing_SemSeg/checkpoints/epoch_018.ckpt",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="传 data 根目录，不是 data/BraDD",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="从哪个 split 取样本可视化",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="总共保存多少个样本",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="可视化时 dataloader 的 batch size",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="BraDD_AnySat_LP",
        help="通常保持 BraDD_AnySat_LP 即可",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./vis_bradd_preds",
        help="输出图片目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--timestep_mode",
        type=str,
        default="last_valid",
        choices=["first_valid", "middle_valid", "last_valid"],
        help="展示哪个有效时相",
    )
    parser.add_argument(
        "--no_prob_map",
        action="store_true",
        help="不保存前景概率图",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # 1) 复用原项目配置
    cfg = build_cfg(args.exp, args.data_dir)

    # 可视化时把 batch 改小一点
    cfg.dataset.global_batch_size = args.batch_size

    # 2) 实例化 datamodule 和 model
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model.instance)

    # 3) 加载 checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    model.to(device)

    # 4) 构建 dataloader
    if args.split in ["train", "val"]:
        datamodule.setup("fit")
    else:
        datamodule.setup("test")

    if args.split == "train":
        loader = datamodule.train_dataloader()
    elif args.split == "val":
        loader = datamodule.val_dataloader()
    else:
        loader = datamodule.test_dataloader()

    # 5) 取若干 batch 做预测并保存
    saved = 0
    rows = []

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            print("label unique:", torch.unique(batch["label"]))
            print("label shape:", batch["label"].shape)
            pred = model(batch)  # [B, C, H, W]

            bsz = pred.shape[0]
            for i in range(bsz):
                row = save_one_sample(
                    batch=batch,
                    pred=pred,
                    out_dir=out_dir,
                    idx=i,
                    timestep_mode=args.timestep_mode,
                    save_prob_map=not args.no_prob_map,
                )
                rows.append(row)
                saved += 1
                if saved >= args.num_samples:
                    save_summary_csv(rows, out_dir)
                    print(f"Saved {saved} samples to: {out_dir}")
                    return

    save_summary_csv(rows, out_dir)
    print(f"Saved {saved} samples to: {out_dir}")


if __name__ == "__main__":
    main()