from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import hydra
import pyrootutils
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


# 定位项目根目录
ROOT = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# 把 src 加入 sys.path，保证能 import data / models
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


def build_cfg(exp_name: str, data_dir: str, batch_size: int) -> DictConfig:
    config_dir = str(Path(ROOT) / "configs")

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


def get_fine_module(lightning_module: torch.nn.Module) -> torch.nn.Module:
    """
    原项目里 hydra 实例化出来的是 LightningModule。
    真正的下游模型通常在 lightning_module.model 里面，也就是 Fine。
    """
    if hasattr(lightning_module, "model"):
        return lightning_module.model
    raise AttributeError("Cannot find `.model` inside instantiated LightningModule.")


@torch.no_grad()
def extract_feature_from_fine(fine_model: torch.nn.Module, batch: dict) -> torch.Tensor:
    """
    提取 head 之前的 frozen backbone feature。

    对 TS2C_AnySat_LP 这种分类任务，一般流程是：
        encoder 输出 tokens: [B, N, D]
        pooling 之后: [B, D]
        head 输入: [B, D]

    这里复刻 Fine.forward() 里 head 之前的逻辑。
    """
    x = fine_model.model(batch)

    if fine_model.global_pool:
        if fine_model.global_pool == "avg":
            feat = x[:, 1:].mean(dim=1)
        elif fine_model.global_pool == "max":
            feat, _ = torch.max(x[:, 1:], dim=1)
        else:
            # token / cls token
            feat = x[:, 0]
        return feat

    # 一般 TS2C 分类不会走到这里；保底处理
    if isinstance(x, tuple):
        x = x[0]
    feat = x[:, 1:]
    return feat


def extract_split(
    split_name: str,
    loader,
    lightning_module: torch.nn.Module,
    device: torch.device,
    out_path: Path,
) -> None:
    fine_model = get_fine_module(lightning_module)

    lightning_module.eval()
    fine_model.eval()

    features = []
    labels = []
    names = []

    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_to_device(batch, device)

            feat = extract_feature_from_fine(fine_model, batch)
            y = batch["label"]

            features.append(feat.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

            if "name" in batch:
                batch_names = batch["name"]
                if isinstance(batch_names, (list, tuple)):
                    names.extend([str(n) for n in batch_names])
                else:
                    names.extend([str(n) for n in batch_names])

            total += feat.shape[0]

            if batch_idx % 50 == 0:
                print(
                    f"[{split_name}] batch={batch_idx}, "
                    f"feature_shape={tuple(feat.shape)}, "
                    f"label_shape={tuple(y.shape)}, "
                    f"total={total}"
                )

    x_all = np.concatenate(features, axis=0).astype("float32")
    y_all = np.concatenate(labels, axis=0)

    # label 可能是 [N, 1]，统一压成 [N]
    if y_all.ndim > 1:
        y_all = y_all.reshape(y_all.shape[0], -1)
        if y_all.shape[1] == 1:
            y_all = y_all[:, 0]

    y_all = y_all.astype("int64")

    save_dict = {
        "x": x_all,
        "y": y_all,
    }

    if len(names) == len(y_all):
        save_dict["name"] = np.array(names, dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save_dict)

    print(f"[DONE] saved {split_name} to {out_path}")
    print(f"       x shape = {x_all.shape}")
    print(f"       y shape = {y_all.shape}")
    print(f"       y unique = {np.unique(y_all)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(ROOT) / "data"),
        help="传 data 根目录，不是 data/TimeSen2Crop",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="TS2C_AnySat_LP",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(ROOT) / "outputs" / "ts2c_features"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    print("[INFO] Building config...")
    cfg = build_cfg(args.exp, args.data_dir, args.batch_size)

    print("[INFO] Instantiating datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    print("[INFO] Instantiating model...")
    lightning_module = hydra.utils.instantiate(cfg.model.instance)
    lightning_module.to(device)
    lightning_module.eval()

    # train / val
    print("[INFO] Setting up fit dataloaders...")
    datamodule.setup("fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    extract_split(
        split_name="train",
        loader=train_loader,
        lightning_module=lightning_module,
        device=device,
        out_path=out_dir / "train_features.npz",
    )

    extract_split(
        split_name="val",
        loader=val_loader,
        lightning_module=lightning_module,
        device=device,
        out_path=out_dir / "val_features.npz",
    )

    # test
    print("[INFO] Setting up test dataloader...")
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    extract_split(
        split_name="test",
        loader=test_loader,
        lightning_module=lightning_module,
        device=device,
        out_path=out_dir / "test_features.npz",
    )

    print("[ALL DONE]")


if __name__ == "__main__":
    main()