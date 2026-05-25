# 本地Windows上对于TimeSen2Crop数据集的smoke测试指令
python src/train.py exp=TS2C_AnySat_LP paths.data_dir=".\data" logger=csv extras.enforce_tags=False max_epochs=1
# 远端服务器上对于TimeSen2Crop数据集的训练指令
python src/train.py exp=TS2C_AnySat_LP paths.data_dir=".\data" logger=csv extras.enforce_tags=False test=False dataset.global_batch_size=1024 trainer.trainer.num_sanity_val_steps=0
# 本地Windows上对于BraDD数据集的smoke测试指令
python src/train.py exp=BraDD_AnySat_LP paths.data_dir=".\data" logger=csv extras.enforce_tags=False test=False dataset.global_batch_size=2 +trainer.trainer.num_sanity_val_steps=0 max_epochs=1
# 远端服务器上对于BraDD数据集的训练指令
python train.py exp=BraDD_AnySat_LP paths.data_dir=/root/autodl-tmp/AnySat-main/data logger=csv extras.enforce_tags=False test=True dataset.global_batch_size=8 +trainer.trainer.num_sanity_val_steps=0
# 模型预测BraDD数据集中部分数据并可视化（本地）
python src\visualization\BraDD_Visualization.py --ckpt_path "logs\BraDD_AnySat_LinearProbing_SemSeg\checkpoints\epoch_018.ckpt" --data_dir "data" --split val --num_samples 8 --batch_size 2 --out_dir "vis_bradd_preds"
# 本地Windows上导出 TimeSen2Crop 的 frozen backbone features
python src/jittor_lp/export_ts2c_features.py --data_dir "./data" --batch_size 128 --out_dir "./src/jittor_lp/outputs/ts2c_features"
# 本地Windows上使用Jittor框架对于TimeSen2Crop数据集的smoke测试指令
# 由于本地Windows在C++编译层面仍有些许问题，因此该指令仍有些问题（可能与操作系统不是Linux有关）
python src/jittor_lp/train_ts2c_head_jittor.py --train_npz outputs/ts2c_features/train_features.npz --val_npz outputs/ts2c_features/val_features.npz --batch_size 128 --epochs 30 --lr 2e-4 --weight_decay 0.05 --limit_train_samples 512 --out_dir outputs/jittor_ts2c_lp_debug
# 远端服务器上使用Jittor框架对于TimeSen2Crop数据集的训练指令
python src/jittor_lp/train_ts2c_head_jittor.py \
  --train_npz /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/ts2c_features/train_features.npz \
  --val_npz /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/ts2c_features/val_features.npz \
  --test_npz /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/ts2c_features/test_features.npz \
  --batch_size 2048 \
  --epochs 200 \
  --lr 2e-4 \
  --weight_decay 0.05 \
  --out_dir /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/jittor_ts2c_lp_full
# 远端服务器上使用pytorch框架对于backbone在TimeSen2Crop数据集上导出的特征向量直接输入到分类头的对照组训练指令（与Jittor框架训练配置尽可能对齐）
python src/jittor_lp/train_ts2c_head_pytorch.py \
  --train_npz /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/ts2c_features/train_features.npz \
  --val_npz /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/ts2c_features/val_features.npz \
  --test_npz /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/ts2c_features/test_features.npz \
  --batch_size 2048 \
  --epochs 200 \
  --lr 2e-4 \
  --weight_decay 0.05 \
  --out_dir /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/pytorch_ts2c_lp_full
# 远端服务器上使用pytorch框架对于backbone在TimeSen2Crop数据集上导出的特征向量直接输入到分类头的对照组训练指令（与原项目训练参数尽可能对齐）
python src/jittor_lp/train_ts2c_head_pytorch.py \
  --train_npz /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/ts2c_features/train_features.npz \
  --val_npz /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/ts2c_features/val_features.npz \
  --test_npz /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/ts2c_features/test_features.npz \
  --batch_size 2048 \
  --epochs 200 \
  --lr 2e-4 \
  --weight_decay 0.05 \
  --scheduler cosine \
  --warmup_epochs 10 \
  --out_dir /root/autodl-tmp/AnySat-main/src/jittor_lp/outputs/pytorch_ts2c_lp_full_cosine