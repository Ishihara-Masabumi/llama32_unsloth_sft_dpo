# Issue #7: unsloth の padding-free とは何か ― with-padding との違い

unsloth が学習開始時にログに出す `🦥 Unsloth: Padding-free auto-enabled, enabling faster training.` の正体について整理する。本リポの SFT/DPO で実際に効いていた最適化の中身。

## ① with-padding(標準のミニバッチ作り方)

ミニバッチを作るとき、配列を一定の長さに揃える必要があるため、短い配列の末尾にダミートークン(`<pad>`)を埋める。

### 例: バッチサイズ 3、各シーケンスの実長が 100 / 800 / 1024 トークン

```
seq1: [t1, t2, ..., t100, PAD, PAD, ..., PAD]   ← 100 real + 924 pad
seq2: [t1, t2, ..., t800, PAD, ..., PAD]         ← 800 real + 224 pad
seq3: [t1, t2, ..., t1024]                       ← 1024 real (paddingなし)

行列としてはこう見える:
┌─────────────────────────────────────────┐
│ ■■■■□□□□□□□□□□□□□□□□□□□□□□□□□□□□│  seq1 (■=real, □=pad)
│ ■■■■■■■■■■■■■■■■■■■■■■■■■□□□□□□│  seq2
│ ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■│  seq3
└─────────────────────────────────────────┘
   ↑ 全て同じ列数 = max_len = 1024 にパッド
```

3D テンソル `[batch=3, seq_len=1024, hidden]` として GPU に流す。

### 問題点

1. **計算の無駄**: PAD トークンも 28 層 attention/FFN を通る。attention mask で attention の値だけは無視されるが、**linear/FFN/RoPE/LayerNorm の forward FLOPs はパディング分も走る**
2. **メモリの無駄**: 各層の activation バッファが `batch × max_len × hidden` で確保される。実トークン総和は 1924 だが、確保されるのは **1024 × 3 = 3072 トークン分の activation**

上の例だと **1148 / 3072 = 37% のパディング**で、その分の compute と activation メモリが純粋に捨てられている。Aratako DPO データのような可変長応答(数百〜数千 tokens)では、**パディング率が 50〜70% になる**ことも珍しくない。

## ② padding-free の仕組み

「**複数シーケンスを末尾連結して 1 本の長い配列にし、attention カーネルに『ここでシーケンスが切れる』という境界情報(cu_seqlens)を渡す**」方式。

### 同じバッチを padding-free で

```
連結:  [t1..t100 | t1..t800 | t1..t1024]   ← 計 1924 tokens、PAD ゼロ
       ↑seq1     ↑seq2      ↑seq3

cu_seqlens = [0, 100, 900, 1924]
              ↑   ↑    ↑    ↑
            seq1 seq2 seq3 end
            開始 開始 開始
```

batch 軸を flatten して **1D の長い配列 [1924, hidden]** にしてしまい、attention カーネル側で境界情報から正しく独立性を保つ。FlashAttention 2 の `varlen_attn` API、または unsloth が内部実装する同等カーネルがこれを処理。

### attention のブロック対角構造

```
       a  b  c  d  | x  y  | p  q  r  s  t
     ┌──────────────┬──────┬──────────────┐
   a │ ●  -  -  -  │ ─  ─ │ ─  ─  ─  ─  ─ │
   b │ ●  ●  -  -  │ ─  ─ │ ─  ─  ─  ─  ─ │
   c │ ●  ●  ●  -  │ ─  ─ │ ─  ─  ─  ─  ─ │   seq1 ブロック
   d │ ●  ●  ●  ●  │ ─  ─ │ ─  ─  ─  ─  ─ │
     ├──────────────┼──────┼──────────────┤
   x │ ─  ─  ─  ─  │ ●  - │ ─  ─  ─  ─  ─ │   seq2 ブロック
   y │ ─  ─  ─  ─  │ ●  ● │ ─  ─  ─  ─  ─ │
     ├──────────────┼──────┼──────────────┤
   p │ ─  ─  ─  ─  │ ─  ─ │ ●  -  -  -  - │
   q │ ─  ─  ─  ─  │ ─  ─ │ ●  ●  -  -  - │
   r │ ─  ─  ─  ─  │ ─  ─ │ ●  ●  ●  -  - │   seq3 ブロック
   s │ ─  ─  ─  ─  │ ─  ─ │ ●  ●  ●  ●  - │
   t │ ─  ─  ─  ─  │ ─  ─ │ ●  ●  ●  ●  ● │
     └──────────────┴──────┴──────────────┘

● = attention 計算する(下三角 causal mask)
─ = attention しない(別シーケンスの境界を超えるので 0)
```

cu_seqlens によって**シーケンスの境界を超えた attention は完全に 0**にされる。だから `a` は `x, p` を見ないし、`p` は `a, x` を見ない。連結はメモリ上の話で、**各シーケンスは独立性を保ったまま学習される**。

## ③ 何がどれだけ変わるか

| 項目 | with-padding | padding-free | 上記例での差 |
| --- | --- | --- | --- |
| **forward FLOPs**(linear, FFN, RoPE) | 3072 tokens 分 | 1924 tokens 分 | **-37%** |
| **attention 計算量** | 3072² の attention(mask で実質減るが allocation は full) | 100² + 800² + 1024² | **大幅減**(全体 attention より小さい) |
| **activation メモリ** | `batch × max_len × hidden × layers` | `total_tokens × hidden × layers` | **-37%** |
| **gradient メモリ** | 同様にパディング込み | 同様にパディング無し | **-37%** |

→ 平均パディング率 30〜50% のバッチで、**compute と memory が 30〜50% 減**。これが unsloth の `Padding-free auto-enabled` ログメッセージの正体。

## ④ padding-free が効くフェーズ・効かないフェーズ

| 段階 | 1 step で処理するトークン数 | パディング | padding-free の効果 |
| --- | --- | --- | --- |
| **訓練** | バッチ全 token 合計(数千〜) | 多い(可変長応答) | **大** |
| **推論 prefill** | バッチ内 prompt 合計(数百〜数千) | 多い(プロンプト長バラつく) | **大** |
| **推論 decode** | バッチサイズと同数(数〜数十) | ほぼ無い(全 seq が 1 token) | **ほぼ無し** |

要するに「**1 step で大量のトークンを処理 + 長さがバラバラ**」のときに最大限効く最適化。訓練と prefill はこの条件にピッタリ当てはまる(詳しくは [Issue #8](https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/8) 参照)。

## ⑤ 「token-level interleaving」との混同に注意

質問でよくあるのが「複数シーケンスをトークン単位で交互に並べるのか?」だが、**そうではない**。

```
❌ NG: [a, x, p, b, y, q, c, p, ...]   ← トークン単位で交互
       これだと attention の独立性が崩れる

✅ OK: [a, b, c, d, x, y, p, q, r, s, t]   ← シーケンス単位で末尾連結
       各シーケンス内の順序は保持される
```

padding-free は**末尾連結方式**であって、**交互挿入は一切しない**。各シーケンス内のトークン順序は元のまま隣り合っている。

## ⑥ 本リポでの実測

unsloth は SFT/DPO 学習開始時に自動的にこれを有効化していた:

```
🦥 Unsloth: Padding-free auto-enabled, enabling faster training.
```

具体的な効果(本リポ実測):

- MCQ-SFT: 1 step あたり ~1.5 秒(plain PEFT+TRL は経験的に ~3 秒)
- DPO: 1 step あたり ~7.4 秒(chosen + rejected + reference の 3 系統 forward)
- VRAM ピーク: DPO で 13GB(理論的にもっと食うはずなのに低い)

詳しい VRAM の話は [Issue #6](https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/6) 参照。

## 関連

- FlashAttention 2 varlen API: https://github.com/Dao-AILab/flash-attention
- Issue #6(VRAM 比較訂正版): https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/6
- Issue #8(prefill / decode の話): https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/8
