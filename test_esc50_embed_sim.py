"""在 ESC-50 上取 2 条同类 + 1 条异类音频，同时输出分类结果和 embedding 余弦相似度。

PaSST.forward 返回 (logits, features)：
  logits    [50]   分类分数
  features  [768]  CLS 与 DIST 的平均，用作 embedding

余弦相似度在 L2 归一化后的 768 维向量上计算，同类应明显高于异类。
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from test_esc50_ckpt import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_CKPT,
    DEFAULT_CSV,
    build_model,
    load_checkpoint,
    load_label_map,
    load_waveform,
)

# fold=1 验证集：两条 dog，一条 thunderstorm（训练时都没见过）
PAIR_SAME = [
    DEFAULT_AUDIO_DIR / "1-100032-A-0.wav",
    DEFAULT_AUDIO_DIR / "1-110389-A-0.wav",
]
PAIR_DIFF = DEFAULT_AUDIO_DIR / "1-101296-A-19.wav"


def infer(net, mel, wave, device):
    """一次前向同时拿到分类概率和 768 维 embedding。"""
    wave = wave.to(device)
    spec = mel(wave.squeeze(1)).unsqueeze(1)
    logits, embedding = net(spec)
    probs = torch.softmax(logits, dim=-1)[0]
    embedding = embedding[0]
    return logits[0].detach().cpu(), probs.detach().cpu(), embedding.detach().cpu()


def main():
    labels, file_meta = load_label_map(DEFAULT_CSV)
    paths = PAIR_SAME + [PAIR_DIFF]
    names = ["同类A", "同类B", "异类C"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, mel = build_model()
    load_checkpoint(net, Path(DEFAULT_CKPT), use_swa=True)
    net.to(device).eval()
    mel.to(device).eval()

    results = []
    print("\n======== 1) 分类结果 ========")
    with torch.no_grad():
        for tag, path in zip(names, paths):
            if not path.exists():
                raise FileNotFoundError(path)
            wave = load_waveform(path)
            logits, probs, embedding = infer(net, mel, wave, device)
            pred_id = int(probs.argmax())
            pred_name = labels[pred_id]
            pred_p = float(probs[pred_id])
            gt_id, gt_name, fold = file_meta[path.name]
            hit = "正确" if pred_id == gt_id else "错误"
            print(
                f"{tag}  {path.name}\n"
                f"     真值: [{gt_id:02d}] {gt_name:<16s} (fold={fold})\n"
                f"     预测: [{pred_id:02d}] {pred_name:<16s}  p={pred_p:.3f}  {hit}\n"
                f"     embedding: shape={tuple(embedding.shape)}  L2={embedding.norm().item():.3f}\n"
                f"     embedding 前 8 维: {embedding[:8].tolist()}"
            )
            results.append(
                {
                    "tag": tag,
                    "path": path,
                    "gt_name": gt_name,
                    "pred_name": pred_name,
                    "embedding": embedding,
                }
            )

    # 先 L2 归一化再做点积 = 余弦相似度；对角线必为 1
    embs = torch.stack([r["embedding"] for r in results], dim=0)
    embs = F.normalize(embs, dim=1)
    sim = embs @ embs.T

    print("\n======== 2) Embedding 余弦相似度 ========")
    header = f"{'':8s}" + "".join(f"{r['tag']:>12s}" for r in results)
    print(header)
    for i, ri in enumerate(results):
        row = f"{ri['tag']:8s}" + "".join(f"{sim[i, j].item():12.3f}" for j in range(len(results)))
        print(row)

    same_ab = sim[0, 1].item()
    diff_ac = sim[0, 2].item()
    diff_bc = sim[1, 2].item()
    print("\n======== 3) 对比结论 ========")
    print(f"同类  {results[0]['gt_name']} vs {results[1]['gt_name']}:  {results[0]['tag']}-{results[1]['tag']} = {same_ab:.3f}")
    print(f"异类  {results[0]['gt_name']} vs {results[2]['gt_name']}:  {results[0]['tag']}-{results[2]['tag']} = {diff_ac:.3f}")
    print(f"异类  {results[1]['gt_name']} vs {results[2]['gt_name']}:  {results[1]['tag']}-{results[2]['tag']} = {diff_bc:.3f}")
    if same_ab > max(diff_ac, diff_bc):
        print("结论: 相同类别的 embedding 余弦相似度明显高于不同类别。")
    else:
        print("结论: 这次抽样里同类相似度没有明显高于异类，可换其他音频再试。")


if __name__ == "__main__":
    main()
