# Issue #6: Issue #5 の訂正版 ― VRAM 比較の基準を 4bit ベースに統一

[Issue #5](https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/5) で書いた高速化の整理について、特に「**VRAM ピークが半分以下**」の比較基準が曖昧だったため、本 Issue で訂正・補強する。

## ① 「unsloth 公式が約 2x」は正しいか?

**Free 版(OSS 版)の中心的な広告は「2x faster」で一貫している**ため、Issue #5 の表現は妥当。ただし数字は条件で揺れるので「**控えめ寄りの丸め**」である点を補足する:

| シナリオ | unsloth 公称 |
| --- | --- |
| Llama-3.2 1B/3B QLoRA SFT (Free 版) | **約 2x** |
| Llama-3.1 8B QLoRA SFT (Free 版) | **約 2x**(条件により 1.8〜2.5x) |
| Llama-3.1 70B QLoRA SFT (Free 版) | **約 2x** |
| Pro / Enterprise 版 | 5x〜30x |

→ 本リポは Free 版利用。「**約 2x**」が妥当な要約。

## ② 「VRAM ピーク半分以下」の比較基準 ― 重要な訂正

Issue #5 では「**4bit + padding-free の組み合わせで VRAM ピークが半分以下**」と書いたが、これは説明が雑だった。**16bit との比較ではなく、4bit 同士の比較**である点を明示する。

正確な分解:

| 構成 | 3B モデル DPO の VRAM ピーク(推定) | 削減元 |
| --- | --- | --- |
| (A) bf16(16bit) + 素 PEFT+TRL + with-padding | **30〜40 GB**(OOM スレスレ) | — |
| (B) **4bit 量子化**(bnb) + 素 PEFT+TRL + with-padding | **25〜30 GB** | (A) → 4bit 化で重み 1/3 |
| (C) 4bit + **unsloth + padding-free + smart offload**(本リポ) | **約 13 GB**(実測) | (B) → unsloth で更に半分 |

(A) → (B) の削減: **`bitsandbytes` の 4bit 量子化**による。unsloth 独自ではない。
(B) → (C) の削減: **unsloth の padding-free + gradient offload** による。これが unsloth の本領。

したがって Issue #5 の「VRAM ピーク半分以下」は、正確には:

> 「**素の 4bit PEFT+TRL 比で、unsloth の padding-free / smart offload を組み合わせると VRAM ピークがさらに半分以下(25〜30 GB → 約 13 GB)になる**」

16bit ベース比較ではなく **4bit ベース比較**。16bit からの全体削減を見ると 3〜4 倍の削減になるが、それは大半が `bitsandbytes` の貢献。

## ③ 「絶対的な要件低下」とは何か

「unsloth を使うか否かで、**学習を回すために必要な GPU グレード自体が変わる**」という意味。具体的には:

| 条件 | 必要 GPU(推定) | コスト感(クラウド時間単価×時間) |
| --- | --- | --- |
| 素の PEFT+TRL bf16 で 3B × 14k × 3epoch DPO | **A100 80GB** 単一、または H100 / V100 multi-GPU | $3〜$5/h × 14h ≈ **$40〜$70** |
| 素の PEFT+TRL 4bit | A100 40GB 単一でぎりぎり | $2/h × 12h ≈ **$24** |
| **unsloth 4bit + padding-free**(本リポ) | **L40S 46GB 単一**で完走 ✅ | $1〜$1.5/h × 6h ≈ **$6〜$9** |

「絶対的な要件低下」=**hardware floor(動かすために最低限必要な GPU グレード)が一段下がる**こと。具体的な含意:

1. **入手性**: A100/H100 は AWS/GCP でも空き待ちが頻発するが、L40S は確保しやすい
2. **コスト**: 上表のとおり 1/5〜1/10 程度
3. **電力**: 700W(A100)→ 350W(L40S)で約半分
4. **個人/中小研究室の射程内に入る**: ローカル LLM ファインチューニングが企業ラボ専有でなくなる

## まとめ表現(訂正版)

> 「unsloth 公式が謳う **約 2x の高速化** が L40S 上でも概ね再現された(SFT 系で 1.8〜2.2 倍、DPO で 1.7〜2.0 倍)。さらに **4bit 量子化(bitsandbytes)で重みメモリを 1/3 に、その上で unsloth の padding-free / smart offload によって activation/gradient メモリをもう半分以下**に削減でき、結果として**単一 L40S(46GB)で 3B モデル × 14k example × 3 epoch の DPO が完走可能**となる。速度の 2x よりも、この**絶対的な hardware 要件の低下**(A100 80GB クラス → L40S クラス)のほうが実用上のインパクトが大きい。」

## 関連

- Issue #4 結論部の高速化記述: https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/4
- Issue #5(原版): https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/5
