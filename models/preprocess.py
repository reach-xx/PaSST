"""Mel 频谱前端：波形 → log-Mel，供 PaSST 当作单通道“图像”使用。

流水线（forward）：
  波形 [B, T]
    → 预加重 FIR（核 [-0.97, 1]）
    → STFT 功率谱 [B, n_fft/2+1, F]
    → Kaldi Mel 滤波 [B, n_mels, F]   默认 n_mels=128
    → log
    → （仅训练）SpecAugment 频率/时间掩蔽
    → 平移缩放归一化

F 是时间帧数，约等于 1 + T / hopsize。
ESC-50 5 秒、sr=32000、hop=320 时 F≈500。
"""

import torch.nn as nn
import torchaudio

import torch

from ba3l.ingredients.ingredient import Ingredient

model_ing = Ingredient("spectrograms")

sz_float = 4  # float32 字节数（历史遗留，本文件未使用）
epsilon = 10e-8  # 数值稳定用的极小量（历史遗留，本文件改用 1e-5）


@model_ing.command
class AugmentMelSTFT(nn.Module):
    """可学习参数几乎没有：窗函数和预加重核都是 buffer，Mel 矩阵每次按 fmin/fmax 现算。

    配置与 AST / kagglebirds 的频谱设定相近，以便对上 AudioSet 预训练分布。
    """

    def __init__(self, n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48, timem=192,
                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1, fmax_aug_range=1000):
        torch.nn.Module.__init__(self)
        # adapted from: https://github.com/CPJKU/kagglebirds2020/commit/70f8308b39011b09d41eb0f4ace5aa7d2b0e806e
        # Similar config to the spectrograms used in AST: https://github.com/YuanGongND/ast

        self.win_length = win_length  # STFT 窗长（采样点）。800/32000=25ms
        self.n_mels = n_mels          # Mel 滤波器个数，也是频谱图的“高度”
        self.n_fft = n_fft            # FFT 点数，频率 bins = n_fft/2+1 = 513
        self.sr = sr                  # 采样率，PaSST 一律 32 kHz
        self.htk = htk
        self.fmin = fmin              # Mel 最低频率
        if fmax is None:
            # 留出 fmax_aug_range 的余量，训练时可以随机上移 fmax
            fmax = sr // 2 - fmax_aug_range // 2
            print(f"Warning: FMAX is None setting to {fmax} ")
        self.fmax = fmax
        self.norm = norm
        self.hopsize = hopsize        # 帧移。320/32000=10ms，大约每秒 100 帧
        # Hann 窗；persistent=False 表示不写入 checkpoint
        self.register_buffer('window',
                             torch.hann_window(win_length, periodic=False),
                             persistent=False)
        assert fmin_aug_range >= 1, f"fmin_aug_range={fmin_aug_range} should be >=1; 1 means no augmentation"
        assert fmin_aug_range >= 1, f"fmax_aug_range={fmax_aug_range} should be >=1; 1 means no augmentation"
        self.fmin_aug_range = fmin_aug_range
        self.fmax_aug_range = fmax_aug_range

        # 预加重 FIR：y[t] = x[t+1] - 0.97 * x[t]，高通，压低频抬高频
        # conv1d 核形状 [out_ch, in_ch, k] = [1, 1, 2]，无 padding 时长度 T→T-1
        self.register_buffer("preemphasis_coefficient", torch.as_tensor([[[-.97, 1]]]), persistent=False)
        # SpecAugment：沿频率/时间轴随机遮一块。freqm=0 或 timem=0 则关闭
        if freqm == 0:
            self.freqm = torch.nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = torch.nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x):
        """x: [B, T] 波形，范围约 [-1, 1]。返回 log-Mel [B, n_mels, F]。"""

        # 预加重。unsqueeze 成 [B, 1, T] 才能 conv1d，再 squeeze 回 [B, T-1]
        x = nn.functional.conv1d(x.unsqueeze(1), self.preemphasis_coefficient).squeeze(1)
        # center=True 会在两端补 n_fft//2，帧数大约 1 + T/hop
        x = torch.stft(x, self.n_fft, hop_length=self.hopsize, win_length=self.win_length,
                       center=True, normalized=False, window=self.window, return_complex=False)
        # stft 在 return_complex=False 时最后一维是实/虚部，平方求和得到功率谱 [B, 513, F]
        x = (x ** 2).sum(dim=-1)  # power mag
        # 训练时随机抖动 Mel 覆盖的频率范围；eval 锁死 fmin/fmax，保证可复现
        fmin = self.fmin + torch.randint(self.fmin_aug_range, (1,)).item()
        fmax = self.fmax + self.fmax_aug_range // 2 - torch.randint(self.fmax_aug_range, (1,)).item()
        if not self.training:
            fmin = self.fmin
            fmax = self.fmax

        # Kaldi Mel 银行：形状 [n_mels, n_fft/2]，再 pad 一列对齐 513 个 FFT bin
        mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(self.n_mels,  self.n_fft, self.sr,
                                        fmin, fmax, vtln_low=100.0, vtln_high=-500., vtln_warp_factor=1.0)
        mel_basis = torch.as_tensor(torch.nn.functional.pad(mel_basis, (0, 1), mode='constant', value=0),
                                    device=x.device)
        # 矩阵乘：把线性频率功率谱压到 128 个 Mel 带  →  [B, 128, F]
        with torch.cuda.amp.autocast(enabled=False):
            melspec = torch.matmul(mel_basis, x)

        melspec = (melspec + 0.00001).log()

        # SpecAugment 只在训练开启，避免验证/推理被随机挖空
        if self.training:
            melspec = self.freqm(melspec)
            melspec = self.timem(melspec)

        # 经验归一化，把 log-Mel 大致拉到 Transformer 习惯的数值范围
        melspec = (melspec + 4.5) / 5.  # fast normalization

        return melspec

    def extra_repr(self):
        return 'winsize={}, hopsize={}'.format(self.win_length,
                                               self.hopsize
                                               )
