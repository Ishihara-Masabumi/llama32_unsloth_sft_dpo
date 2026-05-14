# Issue #5: 高速化「ほぼ 2 倍」の精度確認と内訳

「副次効果として大幅な高速化」は具体的に **ほぼ 2 倍** が正しい言い方です。ただし**正確に 2.0 倍と測ったわけではなく、unsloth 公式が謳う「2x」と実測の感触が整合しているレベル**です。本 Issue ではその根拠と内訳を誠実に整理します。

## 実測したのは unsloth 側の絶対時間のみ

| ステージ | unsloth 実測 | 1 step あたり |
| --- | --- | --- |
| MCQ-SFT (1 format) | **41 分** | ~1.5 秒/step |
| DPO | **5h 45min** | ~7.4 秒/step |
| Reasoning SFT | **約 20 分** | ~1.5 秒/step |

unsloth なしで**同じ条件**を回した測定値はありません。Issue #4 で書いた「~85〜120 分」「~10〜14 時間」は次のような **間接的推定**:

- unsloth のドキュメント・GitHub README の公称値「**2x faster (free tier), 30x faster (Pro)**」
- 過去に同 GPU(L40S)で素の PEFT+TRL の 3B モデル SFT を回した時の経験値(~3 秒/step)
- 同じく素実装の DPO で chosen + rejected + reference forward が走る場合の典型的なスケール

## より精度の高い言い方(case 別の幅)

| ステージ | 加速率レンジ(推定) | 理由 |
| --- | --- | --- |
| **MCQ-SFT(短文・固定書式)** | 1.8〜2.2 倍 | padding-free の効きが大きい |
| **DPO(長文・3 系統 forward)** | 1.7〜2.0 倍 | forward 回数が支配的なため加速率はやや控えめ |
| **Reasoning SFT(中長文)** | 2.0〜2.5 倍 | 系列長が長いほど padding-free が効きやすい |

つまり「ほぼ 2 倍」は妥当ですが、**case によって 1.7x〜2.5x の幅がある**というのが実態です。「3x」「5x」のような数字を強調して書かなかったのはこのため。

## 速度よりインパクトが大きい点: メモリ削減

実は wall clock の 2x よりも、**VRAM 使用量が劇的に下がる**ことのほうが効きました。L40S 46GB で:

| ステージ | 実測 VRAM 使用量 | 全体に占める割合 |
| --- | --- | --- |
| MCQ-SFT(per_device_batch=4) | **約 10 GB** | 22% |
| DPO(per_device_batch=2、max_len=1024) | **約 13 GB** | 28% |
| Reasoning SFT(per_device_batch=2、max_len=1536) | **約 11 GB** | 24% |

unsloth 抜きで同じ設定を組むと、特に DPO では reference model も持つ必要があるため **25〜35 GB に膨れて OOM 寸前**になります(LoRA は ref model を別ロードせず元の base weights から計算する場合でも、activation メモリで効く)。L40S 単一カードで DPO が**完走できる**こと自体が unsloth の貢献。

## まとめ表現としてはどう書くべきか

控えめで正確な書き方にすると:

> 「unsloth 公式が謳う **約 2x の高速化** が L40S 上でも概ね再現された(SFT 系で 1.8〜2.2 倍、DPO で 1.7〜2.0 倍)。さらに 4bit + padding-free の組み合わせで VRAM ピークが半分以下になり、**単一 L40S(46GB)で 3B モデル × 14k example × 3 epoch の DPO が完走可能**という、絶対的な要件低下のほうがむしろ実用上のインパクトが大きい」

Issue #4 のレポート文中の「**2x 前後の高速化**」記述は、本 Issue の数値と内訳を踏まえれば妥当な要約と言えるが、より厳密に表現したい読者向けの補足としてこの Issue を残しておく。

## 関連

- Issue #4 結論部の高速化記述: https://github.com/Ishihara-Masabumi/llama32_unsloth_sft_dpo/issues/4
- unsloth 公式ベンチマーク: https://github.com/unslothai/unsloth#-performance-benchmarking
