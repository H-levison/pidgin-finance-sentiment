# Dataset statistics

| | train_full | train_full_sub | train_dedup | dev | test |
|---|---|---|---|---|---|
| Rows | 5121 | 3486 | 3486 | 1281 | 4154 |
| Negative | 3241 | 2221 | 2287 | 813 | 2326 |
| Neutral | 72 | 42 | 51 | 21 | 431 |
| Positive | 1808 | 1223 | 1148 | 447 | 1397 |
| Neutral % | 1.41 | 1.2 | 1.46 | 1.64 | 10.38 |
| Duplicate texts | 512 | 260 | 0 | 34 | 0 |
| Mean tokens | 24.4 | 24.2 | 24.6 | 24.4 | 19.1 |
| Median tokens | 21 | 21 | 22 | 22 | 16 |
| 95th pct tokens | 50 | 50 | 50 | 49 | 46 |
| Vocabulary size | 10611 | 8534 | 8936 | 4712 | 8724 |
| Contains emoji | 397 | 262 | 275 | 104 | 242 |

## Overlap audit (on the release as distributed)

- Duplicate tweets inside train: 512
- Train tweets also in dev: 291
- Train tweets also in test: 1013
- Dev tweets also in test: 266
- Share of test reachable verbatim from train: 22.29%
- Gold label agreement on the train/test overlap: 87.8% (124 tweets labelled differently in the two splits)
