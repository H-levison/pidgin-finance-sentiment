# Results

## test set

| model | accuracy | macro F1 | weighted F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|---|
| afro-xlmr-base-weighted-aug | 0.696 | 0.489 | 0.660 | 0.782 | 0.035 | 0.650 |
| afro-xlmr-base-weighted-seed42 | 0.694 | 0.475 | 0.654 | 0.781 | 0.000 | 0.644 |
| afro-xlmr-base-weighted-seed43 | 0.697 | 0.478 | 0.658 | 0.785 | 0.000 | 0.650 |
| afro-xlmr-base-weighted-seed44 | 0.700 | 0.483 | 0.663 | 0.785 | 0.000 | 0.665 |

## Published pcm results for context

Muhammad et al. (2023b), Table 7. **Accuracy** on the pcm test split, averaged over 5 runs. Compare these only against our accuracy column, and only against our test rows.

| model | accuracy |
|---|---|
| AfriBERTa-large | 0.623 |
| XLM-R-base | 0.638 |
| AfroXLMR-base | 0.676 |
| mDeBERTaV3-base | 0.662 |
| XLM-T-base | 0.666 |
| XLM-R-large | 0.672 |
| AfroXLMR-large | 0.687 |
