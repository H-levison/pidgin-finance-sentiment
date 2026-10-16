# Final results across 3 seeds (test split)

Runs: afro-xlmr-base-weighted-seed42, afro-xlmr-base-weighted-seed43, afro-xlmr-base-weighted-seed44

| metric | mean | min | max | spread |
|---|---|---|---|---|
| accuracy | 0.697 | 0.694 | 0.700 | 0.006 |
| macro_f1 | 0.479 | 0.475 | 0.483 | 0.008 |
| weighted_f1 | 0.658 | 0.654 | 0.663 | 0.009 |
| f1_negative | 0.784 | 0.781 | 0.785 | 0.003 |
| f1_neutral | 0.000 | 0.000 | 0.000 | 0.000 |
| f1_positive | 0.653 | 0.644 | 0.665 | 0.020 |

Seed-to-seed spread on macro F1 is 0.008. Any difference between models smaller than this is not a real difference.

Note: 3 seeds, not the 5 used by the published AfriSenti baselines.
