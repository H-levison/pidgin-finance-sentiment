# Results

## dev set

| model | accuracy | macro F1 | weighted F1 | F1 neg | F1 neu | F1 pos |
|---|---|---|---|---|---|---|
| tfidf-word-unweighted | 0.664 | 0.368 | 0.608 | 0.779 | 0.000 | 0.325 |
| tfidf-word-balanced | 0.665 | 0.424 | 0.657 | 0.748 | 0.000 | 0.524 |
| tfidf-char-balanced | 0.652 | 0.422 | 0.651 | 0.731 | 0.000 | 0.535 |
| bilstm-random-weighted | 0.618 | 0.417 | 0.635 | 0.715 | 0.018 | 0.520 |
| bilstm-w2v-unweighted | 0.671 | 0.422 | 0.659 | 0.759 | 0.000 | 0.507 |
| bilstm-w2v-weighted | 0.643 | 0.420 | 0.643 | 0.712 | 0.000 | 0.548 |
| roberta-base-weighted | 0.728 | 0.467 | 0.718 | 0.799 | 0.000 | 0.603 |
| xlm-roberta-base-weighted | 0.712 | 0.463 | 0.707 | 0.777 | 0.000 | 0.613 |
| afro-xlmr-base-unweighted | 0.742 | 0.475 | 0.729 | 0.811 | 0.000 | 0.614 |
| afro-xlmr-base-weighted | 0.743 | 0.471 | 0.727 | 0.816 | 0.000 | 0.599 |
| tfidf-word-balanced-leaky * | 0.721 | 0.495 | 0.717 | 0.788 | 0.080 | 0.617 |
| afro-xlmr-base-weighted-leaky * | 0.756 | 0.521 | 0.750 | 0.814 | 0.083 | 0.667 |
| afriberta_base-weighted-afriberta | 0.687 | 0.437 | 0.677 | 0.768 | 0.000 | 0.542 |
| afro-xlmr-base-weighted-seed42 | 0.725 | 0.469 | 0.718 | 0.793 | 0.000 | 0.614 |
| afro-xlmr-base-weighted-seed43 | 0.732 | 0.474 | 0.723 | 0.796 | 0.000 | 0.625 |
| afro-xlmr-base-weighted-seed44 | 0.729 | 0.476 | 0.724 | 0.790 | 0.000 | 0.638 |
| afro-xlmr-base-weighted-sub | 0.748 | 0.511 | 0.742 | 0.812 | 0.077 | 0.645 |

\* trained on the release as distributed, with the 1,013 tweets that also appear in the test split left in. Not a fair comparison with the rows above it, which is the point.

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
