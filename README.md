# OctFractalGen

## 项目概述

本项目实现了一个面向 3D 形状生成的分形自回归框架 `OctFractalGen`。整体思路是：

- 用八叉树表示三维结构；
- 在较浅层级递归预测节点的 split 结构；
- 在叶层预测局部几何的离散 VQ 编码；
- 再借助冻结的预训练 `OctGPT VQVAE` 解码为连续几何，并导出 `.obj` 网格。

当前任务聚焦于 `ShapeNet` 的 `airplane` 单类无条件生成。

---

## 运行建议

### 1. 环境安装

```bash
conda create -n octfractalgen python=3.12
pip install -r requirements.txt
```

---

### 2. 下载数据与模型

从[这里](https://disk.pku.edu.cn/link/AA168CB1A3463E4828BFE18DCA781A496B)下载预训练模型

---

### 3. 训练和生成

#### 训练

```bash
cd OctFractalGen
python main_octfractalgen.py \
  --data_dir data_dir \
  --vqvae_ckpt vqvae_dir \
  --logdir log_dir \
  --model shapenet_vq768_b24 \
  --epochs 100 \
  --warmup_epochs 5 \
  --batch_size 32 \
  --num_workers 16 \
  --lr 1e-4 \
  --min_lr 1e-5 \
  --weight_decay 0.05 \
  --grad_clip 3.0 \
  --patch_size 2048 \
  --vq_mask_ratio_min 0.5 \
  --vq_random_flip 0.1 \
  --vq_remask_stage 0.7 \
  --vq_remask_prob 0.1 \
  --vq_use_bit_pos_emb \
  --vq_cond_injection film \
  --vq_cond_cross_attn_heads 4 \
  --vq_loss_weight 2.0 \
  --vq_buffer_size 64 
```

#### 生成

```bash
cd OctFractalGen
python generate_samples.py \
  --ckpt ckpt_dir \
  --vqvae_ckpt vqvae_dir \
  --output_dir output_dir \
  --model shapenet_vq768_b24 \
  --num_samples 8 \
  --resolution 256
```

#### 仅导出粗八叉树

```bash
python generate_samples.py \
  --ckpt ckpt_dir \
  --vqvae_ckpt vqvae_dir \
  --output_dir output_dir \
  --model shapenet_vq768_b24 \
  --raw_octree
```



