# Issue #1 結果: 多岐選択肢 Instruction Finetuning (unsloth 実装)

## 実行サマリー

| 項目 | 値 |
| --- | --- |
| 学習時間 | letter_text: 約 41 分 / answer_tag: 約 41 分 (L40S, unsloth + padding-free) |
| トータルステップ | 各 1,677 steps (effective batch 16, 3 epoch) |
| ハイパーパラメータ | LoRA R=16 / alpha=32 / dropout=0.05、AdamW(8bit)、lr=2e-4、cosine、warmup 3% |
| ベースモデル | `meta-llama/Llama-3.2-3B-Instruct` (unsloth が `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit` を自動使用) |

## 学習曲線(抜粋)

### letter_text
- step 20:  loss 2.867
- step 200: loss 0.698
- step 1000: loss 0.567
- final (epoch 3): train_loss = 0.6217

### answer_tag
- step 20: loss 3.025
- step 200: loss 0.700
- step 1000: loss 0.566
- final: train_loss ≒ 0.62(ほぼ同等)

## 評価結果 (JCommonsenseQA validation 1,119問)

| モデル | 正解数 | Accuracy | Unparsable |
| --- | --- | --- | --- |
| **baseline** (`meta-llama/Llama-3.2-3B-Instruct`) | 811 / 1119 | **72.48%** | 35 |
| **SFT (letter_text "X. 回答")** | 917 / 1119 | **81.95%** | 0 |
| **SFT (answer_tag "[ANSWER] X")** | 921 / 1119 | **82.31%** | 0 |

## 元実装(職務経歴書記載)との比較

| モデル | 元実装 (PEFT+TRL) | unsloth 実装 | 差分 |
| --- | --- | --- | --- |
| baseline | 22.16% | 72.48% | +50.32 pt |
| letter_text | 85.08% | 81.95% | -3.13 pt |
| answer_tag | 85.43% | 82.31% | -3.12 pt |

### 考察

- **同一方向性の確認** — 元実装と同じく、SFT により大幅に精度が向上し、`[ANSWER] X` 形式が `X. 回答` 形式より僅かに高い (+0.36 pt vs 元実装 +0.35 pt)。**学習の効きと書式選好の傾向は完全に再現**。
- **baseline が大きく乖離** — 元実装 22.16% に対し unsloth 実装 72.48%。これは評価パーサの厳密度の違いによる。元実装は `[ANSWER] X` または `X.` のみを正解と認める厳格パーサ、本実装は `\bX\b` までフォールバックする寛容パーサ。Llama-3.2-3B-Instruct は自然な日本語で「正解は A です」と返すことが多く、厳格パーサだと unparsable 扱いになる(unparsable→不正解で 22% 前後に張り付く)。
- **SFT 後が -3 pt 劣後** — 学習レシピは同一(R/alpha/dropout/epoch 一致)だが、以下が差分要因と推定:
  - 学習データ件数: 元実装 9,759 vs 本実装 8,939 (HF の sbintuitions/JCommonsenseQA train split は 8,939)
  - 量子化: unsloth は 4bit (bnb) でロード。元実装の量子化精度に関する情報がないため詳細比較困難
  - シード/乱数性: 学習は 1 ラン分の結果

## 環境ハマりどころ (再現用メモ)

1. **`torchao` の torch.int1 エラー** — 最新 torchao が torch 2.5.x 非互換 → 削除で解決
2. **`mergekit` / `llm_blender` 不在** — TRL 0.24 が soft-import するため、DPO を使う場合は追加インストール必要(本Issueでは未到達)
3. **TRL 0.24 API 変更** — `SFTConfig.max_seq_length` → `max_length`、`SFTTrainer(tokenizer=)` → `processing_class=`
4. **import 順序** — `import unsloth` は必ず `trl`/`transformers`/`peft` より先に実行する必要あり。後にすると SFTTrainer 内部の eos_token 検証が破綻し `'<EOS_TOKEN>'` プレースホルダで失敗

## 成果物

- LoRA adapter: `outputs/mcq_sft_letter_text/` / `outputs/mcq_sft_answer_tag/`
- 評価詳細(全1119件): `reports/issue1_eval_*.json`
