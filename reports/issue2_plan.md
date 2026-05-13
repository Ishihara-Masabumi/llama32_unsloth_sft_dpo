# Issue #2: DPO による RLHF (unsloth 実装)

## 目的

Issue #1 で得た MCQ-SFT 済モデル(`outputs/mcq_sft_answer_tag`)を初期化点として、Aratako 系日本語 Preference データで DPO を実施し、職務経歴書の数値と比較する。

## データセット

- `Aratako/iterative-dpo-data-for-SimPO-iter2`
- フィールド: `prompt`, `chosen`, `rejected` (string), `chosen_score=5`, `rejected_score=4`
- Train: **14,880** / Eval: **784** (resume と完全一致、HF データ 15,664 件を 14,880/784 で分割)

## ハイパーパラメータ

| 項目 | 値 |
| --- | --- |
| 初期 adapter | `outputs/mcq_sft_answer_tag` (Issue#1 で学習済) |
| 量子化 | 4bit (bitsandbytes) |
| LoRA R/alpha/dropout | 16 / 32 / 0.05 |
| Optimizer | AdamW (8bit) |
| LR | 5e-6, cosine, warmup 3% |
| Beta | 0.1 |
| Effective batch | 16 (per_device=2, grad_accum=8) |
| Epochs | 3 |
| max_length / max_prompt_length | 1024 / 512 |
| seed | 42 |

## コマンドライン

```bash
bash run_issue2_dpo.sh
```

内訳:
1. `prepare_dpo.py` でデータ整形(完了済)
2. baseline(LoRAなし)の Chosen 選好精度
3. SFT 後(`outputs/mcq_sft_answer_tag`)の Chosen 選好精度
4. DPO 学習(SFT adapter を warmstart)
5. DPO 後の Chosen 選好精度

## 評価指標: Chosen 選好精度

eval セット 784件 各 `(prompt, chosen, rejected)` について、モデルが `chosen` に与える条件付き対数尤度が `rejected` のそれより大きい比率。完全一致再現。

## 元実装との比較ターゲット

| 段階 | 元実装 | unsloth 実装 |
| --- | --- | --- |
| オリジナル | 53.06% | _TBD_ |
| SFT 後 | 53.44% | _TBD_ |
| DPO 後 | 54.46% | _TBD_ |

## 依存パッケージ問題のメモ

TRL 0.24 の `DPOTrainer` は依存チェーンに `mergekit` と `llm_blender` がある。`mergekit` は pip で入るが、`llm_blender` は古い `transformers.utils.hub.TRANSFORMERS_CACHE` を import するため transformers 5.x で破綻する。本リポでは **`llm_blender` を uninstall**(soft-import で迂回)+ **`mergekit` のみインストール**で解決した。
