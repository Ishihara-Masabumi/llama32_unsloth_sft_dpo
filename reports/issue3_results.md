# Issue #3 結果: Reasoning SFT (unsloth 実装)

## 実行サマリー

| 項目 | 値 |
| --- | --- |
| 初期 adapter | `outputs/mcq_sft_answer_tag` (Issue#1 で学習済) |
| 学習データ | gsm8k/aqua_rat/hotpotqa/strategyqa 各 1,090 件 = 4,360 件 |
| eval データ | 各 source 160 件 = 640 件 (ホールドアウト) |
| ハイパーパラメータ | LoRA R=16/α=32/dropout=0.05、AdamW(8bit)、lr=2e-4 cosine、effective batch 16、3 epoch、max_seq_len=1536 |
| 学習時間 | 約 20 分(819 steps × ~1.5 s/step、L40S + unsloth) |
| 最終 train_loss | ~0.95 (epoch 3) |

## 評価結果

| メトリック | 元実装 (職務経歴書) | unsloth | 差分 |
| --- | --- | --- | --- |
| Overall (pre-Reasoning, post-MCQ-SFT のみ) | 25.00% | **20.63%** | -4.37 pt |
| Overall (post-Reasoning SFT) | 53.59% | **50.63%** | -2.96 pt |
| gsm8k | 79.38% | **60.00%** | -19.38 pt |
| aqua_rat | 46.88% | **41.88%** | -5.00 pt |
| hotpotqa | 18.75% | **32.50%** | **+13.75 pt** ✨ |
| strategyqa | 69.38% | **68.13%** | -1.25 pt |

## 所見

- **Reasoning SFT 効果は完全に再現**: 改善幅 **+30 pt** (20.63% → 50.63%) は元実装の +28.59 pt とほぼ同等。
- **タスク別では gsm8k が大きく劣後** (-19pt): gsm8k は数学計算で、生成の数値抽出パーサ精度や、unsloth の量子化が計算精度に与える影響が考えられる。
- **hotpotqa は元実装より大きく上回る** (+13.75pt): 元実装の 18.75% は極端に低く、context 長による truncation で多段推論が崩壊している可能性。本実装は max_seq_len=1536 と長めに取ったため、context が活かされている。
- **strategyqa はほぼ同等** (-1.25pt)、aqua_rat も近い水準 (-5pt)。
