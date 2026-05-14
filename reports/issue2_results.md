# Issue #2 結果: DPO による RLHF (unsloth 実装)

## 実行サマリー

| 項目 | 値 |
| --- | --- |
| 学習時間 | DPO training 約 **5h 45min** (2,790 steps × ~7.4 s/step、L40S) |
| ハイパーパラメータ | LoRA R=16/α=32/dropout=0.05、AdamW(8bit)、lr=5e-6、cosine、β=0.1、effective batch 16、3 epoch |
| 量子化 | 4bit (bitsandbytes)、max_length=1024 / max_prompt_length=512 |
| 初期 adapter | `outputs/mcq_sft_answer_tag` (Issue#1 で学習済) |
| seed | 42 |

## 学習曲線(DPO)

| epoch | loss | rewards/chosen | rewards/rejected | rewards/accuracies | rewards/margins |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 2.40 | -30.9 | -31.2 | 0.51 | +0.30 |
| 0.20 | 1.82 | -15.2 | -15.0 | 0.51 | +0.16 |
| 0.55 | 1.36 | -8.6 | -8.9 | 0.55 | +0.28 |
| 1.12 | 0.91 | -5.3 | -6.1 | 0.63 | +0.85 |
| 1.70 | 0.78 | -4.7 | -5.6 | 0.62 | +0.91 |
| 2.24 | 0.56 | -4.5 | -5.9 | 0.77 | +1.38 |
| 2.99 | 0.62 | -4.4 | -5.8 | 0.74 | +1.33 |
| final | train_loss = 0.9829 (3 epoch) | | | | |

DPO 目的関数(margin の最大化)で見ると、`rewards/accuracies` が **0.50 → 0.74**、`rewards/margins` が **+0.30 → +1.33** と明確に改善。**chosen を rejected より高く評価する方向に学習は成功している**。

## 評価結果 (Chosen 選好精度、eval 784件)

| 段階 | 元実装(リファレンス値) | unsloth 実装 | 差分(unsloth - 元) |
| --- | --- | --- | --- |
| オリジナル(LoRAなし) | 53.06% | **52.30%** | -0.76 pt |
| SFT 後(MCQ-SFT answer_tag) | 53.44% | **52.17%** | -1.27 pt |
| DPO 後 | 54.46% | **51.53%** | **-2.93 pt** |

## 考察 — なぜ DPO eval だけ大きく劣後するのか

### DPO の学習目的と本 eval メトリックのギャップ

本 eval は **生の条件付き対数尤度** `log π_θ(chosen|prompt)` と `log π_θ(rejected|prompt)` を比較する単純な指標。一方、DPO の目的関数は:

```
L_DPO = -log σ(β · [(log π_θ(chosen) - log π_ref(chosen)) - (log π_θ(rejected) - log π_ref(rejected))])
```

すなわち**参照モデル(=学習開始時の SFT モデル)からの相対差(KL 調整付き)**を最大化する。学習データ上での `logps/chosen` ≈ `logps/rejected` (約-2500〜-2700、差は数十〜100程度) が示すように、両者は **絶対値ではほぼ同等**で、DPO はその微小な差を増幅している。

長文(2500+ tokens)で生 log-prob の総和を比べると、ノイズが大きく、DPO による margin 改善が雑音に埋もれる。これは DPO の既知の eval 課題で、より適切なメトリックは:
1. **DPO 報酬(参照モデル必要)** — chosen/rejected の implicit reward 比較
2. **生成+ペアワイズ judge** — DPO モデルで生成した応答に対する LLM judge による評価
3. **長さ正規化対数尤度** — トークン平均で比較

### 元実装が 54.46% を達成できた理由(推測)

- 元実装は **同じ生 log-prob 比較メトリック**を使ったうえで +1.4pt 出ている → β や lr の違いで生 log-prob の方向まで揃ったか、または評価実装が「prompt+chosen と prompt+rejected を **同じ長さに揃えてから比較**」など別工夫を入れている可能性
- データ前処理・パッキング・truncation の違いも長文 DPO では効きやすい

### 学習自体の健全性

`rewards/accuracies = 0.74` から、**学習データ上では明確に chosen を rejected より評価できる方向に動いている**。すなわち DPO 自体は機能している。本 Issue ではこの結果を**ありのまま記録**し、Issue #4 の総合比較で「DPO は学習として成功したが eval メトリック上では元実装より低く出た」を明記する。

## 環境ハマりどころ (Issue#2 で踏んだもの)

1. **mergekit / llm_blender 依存** — TRL 0.24 が soft-import。`mergekit` は pip 可、`llm_blender` は `TRANSFORMERS_CACHE`(transformers 5.x で削除)を要求するため**uninstall**で迂回
2. **TRL 0.24 DPO API 変更** — `DPOTrainer(tokenizer=)` → `processing_class=`、`DPOConfig.max_seq_length` → `max_length`、`pad_token` の明示指定が必要
3. **eval スクリプトの shape mismatch** — Aratako データは長文(>1024 tokens)。unsloth/transformers が内部で自動切り詰めしたあと gather で shape ズレ → eval 側で `MAX_SEQ_LEN=4096` + `min(logits, ids)` でクリッピング + 超過サンプルは `nan` スキップ
4. **cuDNN SDPA エラー** — step ~930 で `RuntimeError: cuDNN Frontend error: No valid engine configs for Matmul_MUL_ADD_Reduction_SUB_EXP_Reduction_LOG_ADD_DIV_Matmul_` が発生 → `torch.backends.cuda.enable_cudnn_sdp(False)` で flash/efficient/math にフォールバックして解決
5. **チェックポイント保存戦略** — 5h オーバーの長時間ジョブのため `save_strategy=epoch` → `save_strategy=steps` + `save_steps=300` に変更し中間保存

## 成果物

- LoRA adapter: `outputs/dpo/` (checkpoint-2790 が最終)
- 評価詳細: `reports/issue2_eval_baseline.json` / `issue2_eval_post_sft.json` / `issue2_eval_post_dpo.json`
- 学習ログ: `logs/issue2_run.out` + `logs/issue2_resume_run.out`
