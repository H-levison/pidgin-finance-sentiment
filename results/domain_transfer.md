# Domain transfer: general Pidgin tweets to finance and crypto posts

Checkpoint: `afro-xlmr-base-weighted-seed42`

The middle column is the control. The finance posts come from a forum and are longer than AfriSenti's tweets, so the comparison that matters is **finance against length-matched**, not finance against the full test set. Comparing against the full set would confound topic with post length.

| metric | AfriSenti test (all) | AfriSenti length-matched | finance test | domain effect |
|---|---|---|---|---|
| accuracy | 0.694 | 0.759 | 0.355 | -0.404 |
| macro F1 | 0.475 | 0.503 | 0.304 | -0.199 |
| weighted F1 | 0.654 | 0.742 | 0.272 | -0.470 |

Per class F1:

| class | AfriSenti (all) | length-matched | finance |
|---|---|---|---|
| negative | 0.781 | 0.826 | 0.378 |
| neutral | 0.000 | 0.000 | 0.000 |
| positive | 0.644 | 0.684 | 0.535 |

Three caveats that belong in the report rather than buried here:

1. The sets have different class balances, so **weighted F1 is not comparable across columns**. Macro F1 and the per-class rows are.
2. The finance set is small and labelled by one annotator, so a few points of difference is noise, not a finding.
3. Length is controlled; platform is not. The finance posts are forum replies and the AfriSenti posts are tweets, so part of any remaining gap may be register rather than subject matter. Say so.
