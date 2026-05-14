# Issue #8: LLM 推論の 2 段階構造(prefill / decode)とは何か

「LLM はプロンプトに続けて 1 語ずつ生成する」というユーザー直感は**正しい**。ただし内部実装では **prefill** と **decode** という性質の違う 2 つのフェーズに分かれており、これが各種最適化技術([Issue #7](https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/7) の padding-free や、vLLM の PagedAttention 等)の設計を左右する。

## ① 素朴な実装(KV キャッシュ無し)

例として「**日本の首都は**」というプロンプトに対して「**東京です**」と答えさせる:

```
Step 1: モデルに「日本の首都は」を入力          → "東" を生成
Step 2: モデルに「日本の首都は東」を入力        → "京" を生成
Step 3: モデルに「日本の首都は東京」を入力      → "で" を生成
Step 4: モデルに「日本の首都は東京で」を入力    → "す" を生成
Step 5: モデルに「日本の首都は東京です」を入力  → <EOS>(終了)
```

各 step で「これまでの全文」を毎回モデルに通す。直感通りだが、計算量は致命的に悪い:

- Step 1: 6 トークン分の attention
- Step 2: 7 トークン分
- ...
- Step 5: 10 トークン分
- 合計: 6+7+8+9+10 = **40 トークン分**

→ 生成総トークン数 N に対して **O(N²)**。数千トークン生成だと現実的に動かない。

## ② KV キャッシュという最適化

transformer の各層では、attention のために各トークンから **Key(K)** と **Value(V)** を導出する。重要な性質:

> **過去のトークンの K と V は、次のステップで一切変わらない**(causal mask により、後のトークンは過去の K, V のみ参照)

つまり、Step 1 で計算した「日本の首都は」の各トークンの K, V を**メモリにキャッシュしておけば、Step 2 以降は再計算しなくていい**。

```
Step 1: 「日本の首都は」(6 トークン)を一度に処理
        → 各層で 6 トークン分の K, V を計算してキャッシュ保存
        → 最終層 logits の最後の位置から "東" を生成

Step 2: 「東」(1 トークンだけ)を入力
        → この 1 トークンの K, V を計算してキャッシュに追加
        → attention は「自分の Q × キャッシュ済み 6+1=7 トークンの K, V」で計算
        → "京" を生成

Step 3: 「京」(1 トークンだけ)を入力
        → 同様、キャッシュは 8 トークン分に
        → "で" を生成
...
```

合計コスト: 6 + 1 + 1 + 1 + 1 = **10 トークン分**(素朴版の 1/4)。長文ほど節約効果は大きく、**O(N²) → O(N)** になる。

## ③ ここで生じる非対称構造 ― これが prefill と decode

KV キャッシュを使った結果、「最初の 1 step」と「それ以降」が**性質の違うものになる**:

| | Step 1(最初) | Step 2 以降 |
| --- | --- | --- |
| 入力するトークン数 | プロンプト全長(数百〜数千) | **1 トークンのみ** |
| 計算する K, V の数 | プロンプト全長分 | **1 トークン分のみ** |
| 1 step あたりの処理時間 | 長い(数十〜数百 ms) | 短い(数 ms) |
| GPU の使い方 | 大量の行列演算で full 稼働(**compute-bound**) | 1 トークンの線形変換が大半(**memory-bound**) |

これに名前を付けて分けて呼ぶ:

- **Step 1 を「prefill フェーズ」**(プロンプトでキャッシュを「prefill(事前充填)」する)
- **Step 2 以降を「decode フェーズ」**(1 token ずつ「decode(復号)」していく)

## ④ なぜ prefill だけ padding-free が効くのか

[Issue #7](https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/7) で説明した padding-free は、**「1 step で大量のトークンを処理 + 長さがバラバラ」**のときに効く最適化。

| 段階 | 1 step のトークン数 | パディング | padding-free の効果 |
| --- | --- | --- | --- |
| **訓練** | バッチ全 token 合計(数千〜) | 多い(可変長応答) | **大** |
| **推論 prefill** | バッチ内 prompt 合計(数百〜数千) | 多い(プロンプト長バラつく) | **大** |
| **推論 decode** | バッチサイズと同数(数〜数十) | ほぼ無い(全 seq が 1 token) | **ほぼ無し** |

→ 訓練 forward と prefill は**計算量的にほぼ同じ性質**(複数シーケンスを 1 回で一括処理 + 長さバラバラ)で、だから padding-free が両方に転用できる。
→ decode は**質的に違う**(1 トークンずつ処理 + KV cache を参照)ので、別の最適化(continuous batching、PagedAttention)が必要。

## ⑤ この 2 段階構造に固有名詞はあるか?

**特別な一語の名前は確立されていない**。普通は「**LLM 推論**」「**自己回帰生成**」「**KV キャッシュ付き推論**」と呼ばれ、2 段構造はその実装上の常識として暗黙に含まれている。

### 一般的な呼び方

| 呼び方 | ニュアンス |
| --- | --- |
| **Autoregressive generation / decoding**(自己回帰生成) | 「1 トークンずつ条件付きで生成」を強調 |
| **Causal LM inference** | causal mask を使った言語モデルの推論一般 |
| **KV-cached inference** | KV キャッシュを使った高速推論を強調 |
| **Token-by-token generation** | 動作の見た目を素直に表現 |
| **LLM inference / serving** | 実運用の推論サービス全般 |

### 2 フェーズを意識した呼称(主に serving/最適化の文脈)

- **"prefill phase" / "decode phase"** — そのままフェーズ名で呼ぶ
- **"compute-bound prefill / memory-bound decode"** — それぞれのボトルネック特性で呼び分ける

### 2 フェーズを物理的に分けるアーキテクチャ(2023〜2024 の研究/実装)

prefill と decode の特性が違いすぎるため、**別の GPU で別々に処理する**設計も出てきている。これらには固有名詞がある:

| 名前 | 出所 | 概要 |
| --- | --- | --- |
| **Disaggregated inference** / **Split inference** | 一般用語 | prefill と decode を別マシンに分ける考え方の総称 |
| **Splitwise** | Microsoft Research(2023) | prefill 用 GPU クラスタと decode 用 GPU クラスタを分離 |
| **DistServe** | UC San Diego(2024) | 同上、論文化された disaggregated serving |
| **SplitFuse** / **Chunked Prefill** | DeepSpeed-MII | 長い prefill を chunk に分割して decode と混在実行 |
| **Continuous batching** | Orca → vLLM | decode フェーズで動的にバッチを再構成 |
| **PagedAttention** | vLLM | decode フェーズの KV cache をページ管理 |

これらは「**prefill と decode の特性が違うこと**」を前提として、それを活かす個別の最適化技術。

## ⑥ 訓練/prefill/decode の関係まとめ

| | 名前 | 1 step で扱うトークン | 同じ最適化が使えるか |
| --- | --- | --- | --- |
| 訓練 | **Training forward pass** | 全シーケンス(数百〜数千) | padding-free OK |
| 推論 prefill | **Prefill phase** | プロンプト全体(数百〜数千) | padding-free OK |
| 推論 decode | **Decode phase** | **1 トークンずつ** | 別の最適化(continuous batching 等) |

訓練 forward と prefill は計算量的に同型なので、本リポで使った **unsloth の padding-free は訓練・推論両方で原理的に活かせる**最適化と言える(unsloth のフラグは現状訓練向けに自動有効化されるが、同じカーネルは prefill にも応用可能)。

## ⑦ ユーザーの直感への答え

> ユーザー入力(プロンプト)に続けて生成語を 1 語 1 語足していくのではないですか。

**外見的な動作はその通り**。ただし内部実装では:

1. **「プロンプト全体を最初に 1 回ガッと処理する」**重い 1 step(= prefill)
2. **「1 語生成、1 語生成、1 語生成…」**を繰り返す軽い step たち(= decode)

の 2 つの作業負荷が極端に違うため、実装の最適化や、複数リクエストの同時処理の仕組みも、この 2 段階で別々に設計されている。**「1 語ずつ」という見た目** と **「最初の 1 回だけ重い、あとはずっと軽い」という実装** のギャップが、prefill / decode という呼び分けの動機。

## 関連

- Issue #7(padding-free): https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/7
- vLLM PagedAttention 論文: https://arxiv.org/abs/2309.06180
- Splitwise(Microsoft): https://arxiv.org/abs/2311.18677
- DistServe: https://arxiv.org/abs/2401.09670
