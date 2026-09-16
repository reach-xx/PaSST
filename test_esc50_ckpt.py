"""用 ESC-50 微调得到的 Lightning checkpoint 做单条/批量推理。

默认测 fold=1 验证集里 10 条音频（训练时没见过）。
ESC-50 是 50 类单标签，用 softmax，不要用 AudioSet 的 sigmoid。
"""

from pathlib import Path
import argparse

import librosa
import numpy as np
import pandas as pd
import torch

import helpers.compat  # noqa: F401  # Python 3.12 下先修 sacred 再间接 import 模型
from models.passt import passt_s_swa_p16_128_ap476
from models.preprocess import AugmentMelSTFT

ROOT = Path(__file__).resolve().parent
DEFAULT_CKPT = ROOT / "lightning_logs/version_0/checkpoints/epoch=9-step=9.ckpt"
DEFAULT_CSV = ROOT / "audioset_hdf5s/esc50/meta/esc50.csv"
DEFAULT_AUDIO_DIR = ROOT / "audioset_hdf5s/esc50/audio_32k"
TARGET_SR = 32000  # 必须与预训练 / Mel 前端一致
CLIP_SECONDS = 5   # ESC-50 官方时长

# fold=1 是这次训练的验证折，下面几条都没进训练集
DEMO_FILES = [
    "1-100032-A-0.wav",    # dog
    "1-100038-A-14.wav",   # chirping_birds
    "1-100210-A-36.wav",   # vacuum_cleaner
    "1-101296-A-19.wav",   # thunderstorm
    "1-101336-A-30.wav",   # door_wood_knock
    "1-187207-A-20.wav",   # crying_baby
    "1-17367-A-10.wav",    # rain
    "1-13613-A-37.wav",    # clock_alarm
    "1-28135-A-11.wav",    # sea_waves
    "1-1791-A-26.wav",     # laughing
]


def load_label_map(csv_path: Path):
    """target id → 类别名，以及 filename → (id, 名, fold)。"""
    df = pd.read_csv(csv_path)
    mapping = (
        df.drop_duplicates("target")
        .sort_values("target")
        .set_index("target")["category"]
        .to_dict()
    )
    file_to_meta = {
        row.filename: (int(row.target), row.category, int(row.fold))
        for row in df.itertuples(index=False)
    }
    return mapping, file_to_meta


def load_waveform(path: Path):
    """读音频，单声道 32 kHz，定长 5s，形状 [1, 1, 160000] 供 Mel 使用。"""
    wav, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    need = CLIP_SECONDS * TARGET_SR
    if len(wav) < need:
        wav = np.pad(wav, (0, need - len(wav)))
    else:
        wav = wav[:need]
    return torch.from_numpy(wav).float().view(1, 1, -1)


def build_model():
    """结构必须与微调时一致，但 pretrained=False：权重稍后从 ckpt 灌入。

    img_size=(128, 998) 仍按 10s 建时间位置编码（长度 99），5s 前向时再切片。
    """
    net = passt_s_swa_p16_128_ap476(
        pretrained=False,
        num_classes=50,
        in_chans=1,
        img_size=(128, 998),
        stride=(10, 10),
        u_patchout=0,
        s_patchout_t=10,
        s_patchout_f=5,
    )
    # timem=80：ESC-50 频谱更短，时间掩蔽比 AudioSet 的 192 小
    mel = AugmentMelSTFT(
        n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,
        freqm=48, timem=80, htk=False, fmin=0.0, fmax=None, norm=1,
        fmin_aug_range=10, fmax_aug_range=2000,
    )
    return net, mel


def load_checkpoint(net, ckpt_path: Path, use_swa=True):
    """Lightning ckpt 的 key 带 net. / net_swa. 前缀，SWA 一般更好。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    prefix = "net_swa." if use_swa and any(k.startswith("net_swa.") for k in state) else "net."
    subset = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = net.load_state_dict(subset, strict=False)
    print(f"加载 {ckpt_path}")
    print(f"  epoch={ckpt.get('epoch')}  global_step={ckpt.get('global_step')}  权重前缀={prefix}")
    if missing:
        print("  missing:", missing)
    if unexpected:
        print("  unexpected:", unexpected)
    return ckpt


def predict_one(net, mel, wave, device):
    """wave [1,1,T] → Mel [1,1,128,F] → softmax 概率 [50]。"""
    wave = wave.to(device)
    spec = mel(wave.squeeze(1)).unsqueeze(1)  # 加通道维，频谱当 2D 图
    logits, _ = net(spec)  # 第二个返回值是 768 维 embedding，这里只要分类
    probs = torch.softmax(logits, dim=-1)[0]
    return probs.detach().cpu()


def main():
    parser = argparse.ArgumentParser(description="测试 ESC-50 微调 checkpoint")
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--audio", nargs="*", help="音频路径；不填则用 fold=1 的 10 条示例")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-swa", action="store_true", help="用 net 而不是 net_swa")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    labels, file_meta = load_label_map(DEFAULT_CSV)
    if args.audio:
        audio_paths = [Path(p) for p in args.audio]
    else:
        audio_paths = [DEFAULT_AUDIO_DIR / name for name in DEMO_FILES]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, mel = build_model()
    load_checkpoint(net, ckpt_path, use_swa=not args.no_swa)
    net.to(device).eval()
    mel.to(device).eval()  # 关闭 SpecAugment / Mel 频率抖动

    print("\n推理结果（ESC-50 单标签，softmax）:")
    n_correct = 0
    n_labeled = 0
    with torch.no_grad():
        for path in audio_paths:
            if not path.exists():
                print(f"跳过，找不到: {path}")
                continue
            wave = load_waveform(path)
            probs = predict_one(net, mel, wave, device)
            values, indices = torch.topk(probs, k=min(args.top_k, probs.numel()))
            meta = file_meta.get(path.name)
            gt_txt = ""
            if meta is not None:
                gt_id, gt_name, fold = meta
                n_labeled += 1
                pred_id = int(indices[0])
                hit = pred_id == gt_id
                n_correct += int(hit)
                mark = "OK" if hit else "XX"
                gt_txt = f"  真值: [{gt_id:02d}] {gt_name} (fold={fold})  {mark}"
            print(f"\n{path.name}{gt_txt}")
            for rank, (idx, p) in enumerate(zip(indices.tolist(), values.tolist()), 1):
                print(f"  {rank}. [{idx:02d}] {labels[idx]:20s}  {p:.3f}")

    if n_labeled:
        print(f"\n这 {n_labeled} 条有标签音频的 Top-1: {n_correct}/{n_labeled} = {n_correct / n_labeled:.1%}")


if __name__ == "__main__":
    main()
