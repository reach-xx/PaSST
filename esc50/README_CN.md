# 在 ESC-50 上微调 PaSST（passt_s_swa_p16_128_ap476）

本教程说明如何用 **AudioSet 预训练权重 `passt_s_swa_p16_128_ap476`**（mAP = 0.476）微调 [ESC-50](https://github.com/karolpiczak/ESC-50) 环境声音分类。

ESC-50 共 2000 条 **5 秒** 音频，**50 类**，官方评估是 **5 折交叉验证**。微调时会：

1. 加载 AudioSet 上训好的 Transformer 主干
2. 丢掉原来的 527 类分类头，换成随机初始化的 **50 类头**
3. 用较小学习率、Mixup、Structured Patchout 和 SWA 继续训练

实验入口是仓库根目录的 `ex_esc50.py`。

---

## 1. 环境准备

在仓库根目录、已能跑通 `python ex_esc50.py help` 的环境中操作：

```bash
conda activate passt
cd /root/autodl-tmp/PaSST
python ex_esc50.py help
```

需要 GPU。第一次运行会从 GitHub 下载预训练权重（约几百 MB）：

`https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/passt-s-f128-p16-s10-ap.476-swa.pt`

缓存位置一般在 `~/.cache/torch/hub/`。

---

## 2. 准备 ESC-50 数据

请使用作者提供的 **已重采样到 32 kHz** 的数据包，不要直接用原始 44.1 kHz 的 ESC-50。

```bash
cd /root/autodl-tmp/PaSST
mkdir -p audioset_hdf5s
cd audioset_hdf5s
wget https://github.com/kkoutini/PaSST/releases/download/v.0.0.6/esc50.zip
unzip esc50.zip
```

解压后，下面两个路径必须存在（这是 `esc50/dataset.py` 的默认设置）：

```text
audioset_hdf5s/esc50/meta/esc50.csv
audioset_hdf5s/esc50/audio_32k/
```

可选目录 `audioset_hdf5s/esc50/irs/` 用于冲击响应增强。没有也不影响基本训练。

如果数据不在默认位置，训练时改 `basedataset.base_dir`，例如：

```bash
python ex_esc50.py with basedataset.base_dir=/你的路径/esc50/ ...
```

`base_dir` 必须指向包含 `meta/` 和 `audio_32k/` 的那一层。

检查 csv 是否正常：

```bash
head -n 5 audioset_hdf5s/esc50/meta/esc50.csv
```

csv 里应有 `filename,fold,target,category` 等列。`fold` 为 1–5，`target` 为 0–49。

---

## 3. 默认配置里容易踩的坑

`ex_esc50.py` 已经把下游任务设成 50 类，但 **没有写死骨干网络名字**。`models/passt.py` 里 `get_model()` 的默认 `arch` 是：

```text
passt_s_kd_p16_128_ap486
```

也就是知识蒸馏版，**不是**你要的 `passt_s_swa_p16_128_ap476`。

因此命令行必须显式指定：

```text
models.net.arch=passt_s_swa_p16_128_ap476
```

另外，不要只写命名配置 `passt_s_ap476`。它会整体替换 `models` 字典，可能把 ESC-50 用的 Mel 前端 `models.mel` 覆盖掉。正确做法是只改 `models.net.arch` 这一项。

---

## 4. 微调时权重怎么加载

`pretrained=True`（默认）时会发生：

| 部分 | 行为 |
|---|---|
| Patch Embedding、Transformer、位置编码 | 加载 AudioSet `ap476` 权重 |
| `head` / `head_dist`（原 527 类） | 因类别数变成 50，预训练分类头被丢弃 |
| 新的 50 类头 | 随机初始化，随微调一起更新 |

也就是：**主干迁移，分类头重训**。ESC-50 是单标签分类，损失是 `cross_entropy`，不是 AudioSet 的多标签 BCE。

---

## 5. 推荐训练命令

在仓库根目录执行。下面这条对应官方 README 的设定，并指定了 `ap476` 骨干。

### 5.1 先看配置（不训练）

```bash
python ex_esc50.py print_config with \
  models.net.arch=passt_s_swa_p16_128_ap476 \
  models.net.n_classes=50 \
  models.net.s_patchout_t=10 \
  models.net.s_patchout_f=5 \
  basedataset.fold=1 \
  trainer.precision=16
```

确认这些项：

- `models.net.arch = passt_s_swa_p16_128_ap476`
- `models.net.n_classes = 50`
- `models.net.pretrained = True`
- `basedataset.fold = 1`
- `basedataset.base_dir` 指向真实数据

### 5.2 训练第 1 折（最常用）

```bash
python ex_esc50.py with \
  models.net.arch=passt_s_swa_p16_128_ap476 \
  models.net.n_classes=50 \
  models.net.s_patchout_t=10 \
  models.net.s_patchout_f=5 \
  basedataset.fold=1 \
  trainer.precision=16 \
  trainer.max_epochs=10 \
  lr=0.00001 \
  save_last_n=1 \
  -p
```

参数含义：

| 参数 | 值 | 说明 |
|---|---|---|
| `models.net.arch` | `passt_s_swa_p16_128_ap476` | AudioSet 预训练骨干 |
| `models.net.n_classes` | 50 | ESC-50 类别数 |
| `s_patchout_t` / `s_patchout_f` | 10 / 5 | 结构化 Patchout，省显存、起正则作用 |
| `basedataset.fold` | 1 | 第 1 折作验证，其余 4 折训练 |
| `trainer.precision` | 16 | 混合精度 |
| `trainer.max_epochs` | 10 | 代码默认 10 个 epoch |
| `lr` | 1e-5 | 微调学习率（比 AudioSet 从头训更小） |
| `save_last_n` | 1 | 额外按 epoch 存 checkpoint |
| `-p` |  | 开始前打印完整配置 |

`fold=1` 表示：**用 fold≠1 的 1600 条训练，用 fold=1 的 400 条验证**。

### 5.3 小批量冒烟测试

确认数据和模型能跑通时，加上命名配置 `mini_train`（只跑 5 个 batch）：

```bash
python ex_esc50.py with mini_train \
  models.net.arch=passt_s_swa_p16_128_ap476 \
  models.net.s_patchout_t=10 \
  models.net.s_patchout_f=5 \
  basedataset.fold=1 \
  trainer.precision=16 \
  -p
```

### 5.4 官方 5 折交叉验证

ESC-50 论文指标是 5 折准确率平均。依次跑：

```bash
for fold in 1 2 3 4 5; do
  python ex_esc50.py with \
    models.net.arch=passt_s_swa_p16_128_ap476 \
    models.net.n_classes=50 \
    models.net.s_patchout_t=10 \
    models.net.s_patchout_f=5 \
    basedataset.fold=${fold} \
    trainer.precision=16 \
    trainer.max_epochs=10 \
    lr=0.00001 \
    save_last_n=1 \
    -p -c "ESC50 ap476 fold${fold}"
done
```

把每折日志里的 `acc`（以及 `swa_acc`，如果启用了 SWA）记下来，再取平均。作者发布的 ESC-50 微调模型大约 **96.7%**。

---

## 6. 默认训练细节（代码里已经设好）

这些一般不用改，了解即可：

| 项 | 默认 | 说明 |
|---|---|---|
| 优化器 | AdamW，`weight_decay=1e-4` | `get_optimizer` |
| 学习率 | `1e-5` | 微调专用 |
| epoch | 10 | ESC-50 很小 |
| batch size | 训练 12 / 验证 20 | 显存不够就改 `datasets.training.batch_size` |
| Mixup | 开，`mixup_alpha=0.3` | 频谱图级混合 |
| 时间滚动 | `roll=True` | 波形沿时间轴滚动 |
| 音量增强 | `gain_augment=7` | 训练折开启 |
| Mel | 128 bins，32 kHz，hop=320 | 与预训练一致 |
| SpecAugment | `freqm=48`，`timem=80` | 5 秒音频，时间掩蔽比 AudioSet 短 |
| SWA | 第 2 个 epoch 开始，每 epoch 一次 | 验证时会多一个 `swa_acc` |

关闭 Mixup：

```bash
python ex_esc50.py with nomixup models.net.arch=passt_s_swa_p16_128_ap476 ...
```

减小 batch：

```bash
python ex_esc50.py with datasets.training.batch_size=8 models.net.arch=passt_s_swa_p16_128_ap476 ...
```

改数据进程数（机器核少时建议关掉默认 16）：

```bash
python ex_esc50.py with datasets.training.num_workers=4 datasets.test.num_workers=4 ...
```

---

## 7. 训练过程中会看到什么

1. `Loading PASST TRAINED ON AUDISET`：正在下/加载 `ap476`。
2. 分类头因 50 ≠ 527 被丢弃，这是预期行为。
3. `Dataset training fold 1 ... remains 1600` / `testing ... remains 400`。
4. 可能出现：

```text
Input image size (128*500) doesn't match model (128*998)
```

ESC-50 是 5 秒，预训练按约 10 秒（998 帧）设计。PaSST 会插值时间位置编码，**警告可忽略**。

5. 验证指标：
   - `acc`：当前网络准确率
   - `swa_acc`：SWA 平均权重准确率（通常更好，报结果时优先看它）

日志和 checkpoint 默认在：

```text
lightning_logs/version_*/checkpoints/
```

设置了 `save_last_n=1` 时还会按 `step`（epoch）再存一份。

---

## 8. 只评估、不训练

```bash
python ex_esc50.py evaluate_only with \
  models.net.arch=passt_s_swa_p16_128_ap476 \
  models.net.s_patchout_t=10 \
  models.net.s_patchout_f=5 \
  basedataset.fold=1 \
  trainer.precision=16 \
  -p
```

这会用当前随机初始化的 50 类头做验证，分数会很差。要评估微调后的模型，需要自己把 checkpoint 加载进 `M.net`（本仓库的 `evaluate_only` 默认不会读你保存的 ckpt 路径）。更直接的做法是用下面第 9 节的推理脚本，或在 Lightning 里 `trainer.fit` 结束后看最后一轮 `acc` / `swa_acc`。

---

## 9. 用微调后的权重做推理

作者在 [releases v.0.0.6](https://github.com/kkoutini/PaSST/releases/tag/v.0.0.6) 提供了 ESC-50 上训好的权重，例如：

`esc50-passt-s-n-f128-p16-s10-fold1-acc.967.pt`

也可以改成你自己 `lightning_logs` 里的 checkpoint（注意 Lightning ckpt 的 key 通常带 `net.` 前缀，需要按实际情况 `load_state_dict`）。

最小推理示例：

```python
import torch
from hear21passt.base import get_basic_model, get_model_passt

model = get_basic_model(mode="logits")
model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=50)

state_dict = torch.load("你的权重.pt", map_location="cpu")
model.net.load_state_dict(state_dict)

model.eval()
model = model.cuda()

# 输入: [batch, 秒数 * 32000]，32 kHz 单声道，范围约 [-1, 1]
# ESC-50 每条 5 秒 -> 160000 采样点
audio_wave = torch.zeros(1, 32000 * 5).cuda()
with torch.no_grad():
    logits = model(audio_wave)          # [1, 50]
    pred = logits.softmax(dim=-1).argmax(dim=-1)
print(pred)
```

ESC-50 是单标签，推理用 **softmax + argmax**，不要用 AudioSet 那套 sigmoid 多标签。

---

## 10. 常见问题

**Q: 为什么必须写 `models.net.arch=passt_s_swa_p16_128_ap476`？**  
A: 否则会加载默认的 `passt_s_kd_p16_128_ap486`。

**Q: 显存不够？**  
A: 加大 Patchout，或减小 batch：

```text
models.net.s_patchout_t=20 models.net.s_patchout_f=6
datasets.training.batch_size=4
trainer.precision=16
```

**Q: `trainer.gpus` 过时警告？**  
A: 当前环境是 pytorch-lightning 1.9，`gpus=1` 仍可用。若升级到 2.x，需改成 `trainer.devices=1`。

**Q: 要不要 wandb？**  
A: `ex_esc50.py` 的 `main()` 没有启用 WandbLogger，不登录也能训。

**Q: 数据加载很慢 / 报共享内存错误？**  
A: 把 `num_workers` 降到 2 或 4。

**Q: 我想换数据根目录，但找不到 csv。**  
A: `base_dir` 要带最后的 `esc50/`，并且该目录下直接有 `meta/esc50.csv`。

---

## 11. 一条龙最短路径

```bash
conda activate passt
cd /root/autodl-tmp/PaSST

# 1) 数据（只需做一次）
mkdir -p audioset_hdf5s && cd audioset_hdf5s
wget https://github.com/kkoutini/PaSST/releases/download/v.0.0.6/esc50.zip
unzip -n esc50.zip
cd ..

# 2) 确认路径
test -f audioset_hdf5s/esc50/meta/esc50.csv && echo "数据 OK"

# 3) 微调 fold 1
python ex_esc50.py with \
  models.net.arch=passt_s_swa_p16_128_ap476 \
  models.net.n_classes=50 \
  models.net.s_patchout_t=10 \
  models.net.s_patchout_f=5 \
  basedataset.fold=1 \
  trainer.precision=16 \
  trainer.max_epochs=10 \
  lr=0.00001 \
  save_last_n=1 \
  -p
```

训练结束后看验证日志中的 **`swa_acc`** 或 **`acc`**。要报 ESC-50 官方分数，把 fold 1–5 跑完再平均。
