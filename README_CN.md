# PaSST：基于 Patchout 的音频 Transformer 高效训练

本仓库是论文 [Efficient Training of Audio Transformers with Patchout](https://arxiv.org/abs/2110.05069) 的实现。

Patchout 能显著降低在音频频谱图上训练 Transformer 所需的训练时间和 GPU 显存，同时提升模型性能。

<p align="center"><img src="https://github.com/kkoutini/PaSST/blob/main/.github/speed_mem_map.png?raw=true" width="600"/></p>

Patchout 的做法是在训练过程中丢弃部分输入 patch。
丢弃方式既可以是非结构化的（随机丢弃，类似 dropout），
也可以按整段时间帧或频率 bin 丢弃已提取的 patch（类似 SpecAugment），
对应下图第 3 步中的行/列。

<p align="center"><img src="https://github.com/kkoutini/PaSST/raw/main/.github/passt_diag.png?raw=true" width="600"/></p>

## 目录

- [用于推理与嵌入提取的预训练模型](#用于推理与嵌入提取的预训练模型)
  - [从预训练模型获取 logits](#从预训练模型获取-logits)
  - [获取用于微调的预训练模型](#获取用于微调的预训练模型)
- [开发环境](#开发环境)
  - [搭建论文实验所用的开发环境](#搭建论文实验所用的开发环境)
  - [使用导出的 conda 环境搭建](#使用导出的-conda-环境搭建)
  - [检查环境](#检查环境)
- [快速开始](#快速开始)
  - [基本说明](#基本说明)
  - [配置实验](#配置实验)
- [在 Audioset 上训练](#在-audioset-上训练)
- [预训练模型示例](#预训练模型示例)
- [下游数据集微调示例](#下游数据集微调示例)
- [引用](#引用)
- [联系方式](#联系方式)

## 用于推理与嵌入提取的预训练模型

如果你只需要使用预训练模型生成的嵌入、使用自己的微调框架，或者仅用于推理，可以查看本仓库的精简版本 [here](https://github.com/kkoutini/passt_hear21)。
该包遵循 [HEAR 2021 NeurIPS Challenge](https://neuralaudio.ai/hear2021-results.html) API，可通过以下命令安装：

```shell
pip install hear21passt
```

本仓库是完整框架，可用于从头训练模型，以及将在 Audioset 上预训练的模型微调到下游任务。

### 从预训练模型获取 logits

```python
from hear21passt.base import get_basic_model,get_model_passt
import torch
# get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
model = get_basic_model(mode="logits")
print(model.mel) # Extracts mel spectrogram from raw waveforms.
print(model.net) # the transformer network.

# example inference
model.eval()
model = model.cuda()
with torch.no_grad():
    # audio_wave has the shape of [batch, seconds*32000] sampling rate is 32k
    # example audio_wave of batch=3 and 10 seconds
    audio = torch.ones((3, 32000 * 10))*0.5
    audio_wave = audio.cuda()
    logits=model(audio_wave) 
```

### 获取用于微调的预训练模型

```python
from hear21passt.base import get_basic_model,get_model_passt
import torch
# get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
model = get_basic_model(mode="logits")
print(model.mel) # Extracts mel spectrogram from raw waveforms.

# optional replace the transformer with one that has the required number of classes i.e. 50
model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476",  n_classes=50)
print(model.net) # the transformer network.


# now model contains mel + the transformer pre-trained model ready to be fine tuned.
# It's still expecting input of the shape [batch, seconds*32000] sampling rate is 32k

model.train()
model = model.cuda()

```

## 开发环境

如果希望使用与论文相同的环境，可以按照下面的说明操作。

### 搭建论文实验所用的开发环境

若要从头训练模型，或使用与论文相同的配置进行微调：

1. 如有需要，创建一个 Python 3.8 的新环境并激活：

```bash
conda create -n passt python=3.8
conda activate passt
 ```

1. 安装适合你系统的 PyTorch 构建版本。例如：

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

 ```

1. 安装依赖：

 ```bash
pip install -r requirements.txt
 ```

### 使用导出的 conda 环境搭建

也可以使用导出的 conda 环境文件 `environment.yml` 来创建环境。

搭建时建议使用 [Mamba](https://github.com/mamba-org/mamba)，因为它比 `conda` 更快：

```shell
conda install mamba -n base -c conda-forge
```

现在可以从 `environment.yml` 导入环境：

```shell
mamba env create -f environment.yml
```

完成后会得到一个名为 `ba3l` 的环境。

### 检查环境

为了检查你的环境是否与我们实验所用环境一致，请对照 `environment.yml` 和 `pip_list.txt`。这两个文件由以下命令导出：

```shell
conda env export --no-builds | grep -v "prefix" > environment.yml
pip list > pip_list.txt
```

## 快速开始

如果你只想使用自己的训练流程，并仅从本仓库获取模型，可以按上文 [用于推理与嵌入提取的预训练模型](#用于推理与嵌入提取的预训练模型) 所述，从头训练或在自己的数据集上微调。本节其余部分说明如何使用本仓库进行训练和微调。为此，首先需要按上文说明搭建开发环境。

### 基本说明

本仓库使用 [sacred](https://sacred.readthedocs.io/en/) 管理实验与配置，使用 pytorch-lightning 进行训练，使用 wandb 进行日志记录。

每个数据集都有一个主实验文件（如 `ex_audioset.py` 和 `ex_openmic.py`）以及对应的数据集目录。实验文件包含主要的训练与验证逻辑。数据集目录包含下载、预处理以及为训练加载该数据集所需的代码。

一般来说，可以查询实验文件的帮助信息，这会打印可用命令和基本选项：

```shell
python ex_audioset.py help
```

### 配置实验

每个实验都有一组默认配置，定义在实验文件中，例如 `ex_audioset.py`。你可以使用 [sacred 语法](https://sacred.readthedocs.io/en/stable/command_line.html) 覆盖任意配置。可以使用 `print_config` 命令在不训练模型的情况下打印配置值：

```shell
 python ex_audioset.py print_config
 ```

然后可以通过命令行接口覆盖任意配置项（[sacred 语法](https://sacred.readthedocs.io/en/stable/command_line.html)），使用 `with`，例如：

```shell
python ex_audioset.py with trainer.precision=16 
```

这会使用 16 位精度在 Audioset 上训练。

整体配置大致如下：

```yaml
  ...
  seed = 542198583                  # the random seed for this experiment
  slurm_job_id = ''
  speed_test_batch_size = 100
  swa = True
  swa_epoch_start = 50
  swa_freq = 5
  use_mixup = True
  warm_up_len = 5
  weight_decay = 0.0001
  basedataset:
    base_dir = 'audioset_hdf5s/'     # base directory of the dataset, change it or make a link
    eval_hdf5 = 'audioset_hdf5s/mp3/eval_segments_mp3.hdf'
    wavmix = 1
    ....
    roll_conf:
      axis = 1
      shift = None
      shift_range = 50
  datasets:
    test:
      batch_size = 20
      dataset = {CMD!}'/basedataset.get_test_set'
      num_workers = 16
      validate = True
    training:
      batch_size = 12
      dataset = {CMD!}'/basedataset.get_full_training_set'
      num_workers = 16
      sampler = {CMD!}'/basedataset.get_ft_weighted_sampler'
      shuffle = None
      train = True
  models:
    mel:
      freqm = 48
      timem = 192
      hopsize = 320
      htk = False
      n_fft = 1024
      n_mels = 128
      norm = 1
      sr = 32000
      ...
    net:
      arch = 'passt_s_swa_p16_128_ap476'
      fstride = 10
      in_channels = 1
      input_fdim = 128
      input_tdim = 998
      n_classes = 527
      s_patchout_f = 4
      s_patchout_t = 40
      tstride = 10
      u_patchout = 0
      ...
  trainer:
    accelerator = None
    accumulate_grad_batches = 1
    amp_backend = 'native'
    amp_level = 'O2'
    auto_lr_find = False
    auto_scale_batch_size = False
    ...
```

许多配置都可以从命令行更新。
简要说明如下：

- `trainer` 下的所有配置项对应 pytorch lightning trainer 的 [API](https://pytorch-lightning.readthedocs.io/en/1.4.1/common/trainer.html#trainer-class-api)。例如，要关闭 CUDA benchmarking，在命令行中加入 `trainer.benchmark=False`。
- `wandb` 是 wandb 相关配置。例如，要更改 wandb 项目名，在命令行中加入 `wandb.project="test_project"`。
- `models.net` 是 PaSST（或所选神经网络）的选项。例如：`models.net.u_patchout`、`models.net.s_patchout_f`、`models.net.s_patchout_t` 分别控制非结构化 patchout，以及频率维和时域上的结构化 patchout。`input_fdim` 和 `input_tdim` 是输入频谱图在频率维和时间维上的尺寸。`models.net.fstride` 和 `models.net.tstride` 是输入 patch 在频率维和时间维上的步长，设为 16 表示 patch 之间没有重叠。
- `models.mel` 是预处理选项（mel 频谱图）。`mel.sr` 是采样率，`mel.hopsize` 是 STFT 窗的 hop size，`mel.n_mels` 是 mel 滤波器组数量，`mel.freqm` 和 `mel.timem` 是 SpecAugment 的频率掩蔽和时间掩蔽参数。

`config_updates.py` 中有许多预定义的配置包（称为 named_configs），包括不同模型、实验设置等。
可以用以下命令列出这些配置：

```shell
python ex_audioset.py print_named_configs
```

例如，`passt_s_20sec` 是一个配置包，将模型设为在 Audioset 上预训练的 PaSST-S，并可接受最长 20 秒的音频片段。

## 在 Audioset 上训练

请按 [audioset 页面](audioset/) 中的说明下载并准备数据集。

基础 PaSST 模型可以这样训练：

```bash
python ex_audioset.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384 -p
```

例如，仅使用大小为 400 的非结构化 patchout：

```bash
python ex_audioset.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384  models.net.u_patchout=400  models.net.s_patchout_f=0 models.net.s_patchout_t=0 -p
```

可以通过设置环境变量 `DDP` 启用多 GPU 训练，例如使用 2 块 GPU：

```shell
 DDP=2 python ex_audioset.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384 -p -c "PaSST base 2 GPU"
```

## 预训练模型示例

请查看 [releases 页面](https://github.com/kkoutini/PaSST/releases/) 下载预训练模型。
一般来说，可以用下面的方式获取在 Audioset 上预训练的模型：

```python
from models.passt import get_model
model  = get_model(arch="passt_s_swa_p16_128_ap476", pretrained=True, n_classes=527, in_channels=1,
                   fstride=10, tstride=10,input_fdim=128, input_tdim=998,
                   u_patchout=0, s_patchout_t=40, s_patchout_f=4)
```

这会自动下载在 Audioset 上预训练、mAP 为 ```0.476``` 的 PaSST。该模型训练时使用了 ```s_patchout_t=40, s_patchout_f=4```，但你可以修改这些参数，以更好地适配你的任务/计算需求。

还有若干预训练模型可用，它们具有不同的步长（重叠程度），以及是否使用 SWA：`passt_s_p16_s16_128_ap468, passt_s_swa_p16_s16_128_ap473, passt_s_swa_p16_s14_128_ap471, passt_s_p16_s14_128_ap469, passt_s_swa_p16_s12_128_ap473, passt_s_p16_s12_128_ap470`。
例如，在 `passt_s_swa_p16_s16_128_ap473` 中：`p16` 表示 patch 大小为 `16x16`，`s16` 表示无重叠（stride=16），128 个 mel 频带，`ap473` 表示该模型在 Audioset 上的性能 mAP=0.479。

一般来说，可以用以下方式获取预训练模型：

```python
from models.passt import get_model
passt = get_model(arch="passt_s_swa_p16_s16_128_ap473", fstride=16, tstride=16)
```

使用本框架时，可以用以下命令评估该模型：

```shell
python ex_audioset.py evaluate_only with  trainer.precision=16  passt_s_swa_p16_s16_128_ap473 -p
```

也提供了这些模型的集成版本：
一个较大的集成模型，结果为 `mAP=.4956`

```shell
python ex_audioset.py evaluate_only with  trainer.precision=16 ensemble_many
```

由 `stride=14` 和 `stride=16` 两个模型组成的集成，结果为 `mAP=.4858`

```shell
python ex_audioset.py evaluate_only with  trainer.precision=16 ensemble_s16_14
```

还有其他集成：`ensemble_4`、`ensemble_5`

## 下游数据集微调示例

1. [ESC-50：环境声音分类数据集](esc50/)
2. [OpenMIC-2018 数据集](openmic/)
3. [FSD50K](fsd50k/)

## 引用

Interspeech 2022 录用论文的引用格式：

```bib
@inproceedings{koutini22passt,
  author       = {Khaled Koutini and
                  Jan Schl{\"{u}}ter and
                  Hamid Eghbal{-}zadeh and
                  Gerhard Widmer},
  title        = {Efficient Training of Audio Transformers with Patchout},
  booktitle    = {Interspeech 2022, 23rd Annual Conference of the International Speech
                  Communication Association, Incheon, Korea, 18-22 September 2022},
  pages        = {2753--2757},
  publisher    = {{ISCA}},
  year         = {2022},
  url          = {https://doi.org/10.21437/Interspeech.2022-227},
  doi          = {10.21437/Interspeech.2022-227},
}
```

## 联系方式

本仓库会持续更新。如有任何问题，欢迎在 GitHub 上提交 issue，或直接联系作者。
