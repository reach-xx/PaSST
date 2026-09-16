"""使用 PaSST 对 audio.flac 做 AudioSet 事件分类（多标签）。"""

from pathlib import Path
import argparse
import csv

import librosa
import soundfile as sf
import torch

from hear21passt.base import get_scene_embeddings, load_model

TARGET_SR = 32000
ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO = ROOT / "audio.flac"
DEFAULT_LABELS = ROOT / "class_labels_indices.csv"


def load_labels(path: Path):
    labels = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(row["display_name"].strip('"'))
    return labels


def load_audio(path: Path, target_sr: int = TARGET_SR):
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav.mean(axis=1).astype("float32")
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav = torch.from_numpy(wav).clamp(-1.0, 1.0).unsqueeze(0)
    return wav, target_sr


def main():
    parser = argparse.ArgumentParser(description="PaSST AudioSet 事件分类")
    parser.add_argument("audio", nargs="?", default=str(DEFAULT_AUDIO), help="音频路径，默认 audio.flac")
    parser.add_argument("--top-k", type=int, default=10, help="打印概率最高的 K 个事件")
    parser.add_argument("--threshold", type=float, default=0.2, help="认为该事件存在的概率阈值")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"找不到音频文件: {audio_path}")
    if not DEFAULT_LABELS.exists():
        raise FileNotFoundError(f"找不到类别表: {DEFAULT_LABELS}")

    labels = load_labels(DEFAULT_LABELS)
    audio, sr = load_audio(audio_path)
    duration = audio.shape[1] / sr
    print(f"音频: {audio_path}")
    print(f"时长: {duration:.2f} s, 采样率: {sr} Hz, 波形: {tuple(audio.shape)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(mode="logits").to(device)
    model.eval()

    with torch.no_grad():
        # 超过 10 秒时会按 10 秒窗滑动后取平均；本文件约 10 秒，直接整段推理
        logits = get_scene_embeddings(audio, model)
        probs = torch.sigmoid(logits)[0].cpu()

    k = min(args.top_k, probs.numel())
    values, indices = torch.topk(probs, k=k)
    print(f"\nAudioSet 事件分类 Top-{k}（多标签，概率为 sigmoid(logits)）:")
    for rank, (idx, p) in enumerate(zip(indices.tolist(), values.tolist()), 1):
        mark = "*" if p >= args.threshold else " "
        print(f"{rank:2d}.{mark} {labels[idx]:40s}  {p:.3f}")

    detected = [
        (labels[i], float(probs[i]))
        for i in torch.argsort(probs, descending=True).tolist()
        if float(probs[i]) >= args.threshold
    ]
    print(f"\n高于阈值 {args.threshold:.2f} 的事件共 {len(detected)} 个。")


if __name__ == "__main__":
    main()
