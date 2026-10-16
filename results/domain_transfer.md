# Domain transfer: general Pidgin tweets to finance and crypto posts

Checkpoint: `afro-xlmr-base-weighted-seed42`

The middle column is the control. The finance posts come from a forum and are longer than AfriSenti's tweets, so the comparison that matters is **finance against length-matched**, not finance against the full test set. Comparing against the full set would confound topic with post length.

| metric | AfriSenti test (all) | AfriSenti length-matched | finance test | domain effect |
|---|---|---|---|---|
| accuracy | 0.694 | 0.759 | 0.339 | -0.420 |
| macro F1 | 0.475 | 0.503 | 0.296 | -0.207 |
| weighted F1 | 0.654 | 0.742 | 0.254 | -0.489 |

Per class F1:

| class | AfriSenti (all) | length-matched | finance |
|---|---|---|---|
| negative | 0.781 | 0.826 | 0.357 |
| neutral | 0.000 | 0.000 | 0.000 |
| positive | 0.644 | 0.684 | 0.530 |

Three caveats that belong in the report rather than buried here:

1. The sets have different class balances, so **weighted F1 is not comparable across columns**. Macro F1 and the per-class rows are.
2. The finance set is small and labelled by one annotator, so a few points of difference is noise, not a finding.
3. Length is controlled; platform is not. The finance posts are forum replies and the AfriSenti posts are tweets, so part of any remaining gap may be register rather than subject matter. Say so.

## Decomposing the 42-point drop

The headline number confounds two different things, and separating them is the
point of this section.

| component | size | what it is |
|---|---|---|
| the blind spot meeting the domain | about 26 points | neutral is 46.8% of the finance set against 4.3% of the length-matched AfriSenti sample, and the model never predicts neutral, so its ceiling here is 58/109 = 53.2% before it classifies anything |
| genuine domain difficulty | 15.5 points | with neutral removed from both sides, accuracy falls from 0.793 to 0.638 |

Both are real and they say different things. The first is a property of the
training data meeting a different world. The second is the model actually being
worse at reading Pidgin about money, on exactly the two classes it can handle.

## The mechanism: the model imports its training prior

| | predicts negative | actually negative |
|---|---|---|
| AfriSenti, length-matched | 66.5% | 61.7% |
| finance test | **60.6%** | **16.5%** |

On AfriSenti the model's negative rate tracks the truth closely, which is what a
well-calibrated classifier does. On the finance set it keeps predicting negative
at roughly the same rate while the true rate has fallen by a factor of four.

It has not learned "what negative Pidgin looks like" so much as "most Pidgin
tweets are negative". Nigerian crypto forum discussion is mostly people
explaining how wallets and exchanges work, so that prior is wrong, and the model
carries it over unchanged.

This is the clearest single answer to the project's second research question.
A model trained on general Pidgin Twitter does not transfer to Pidgin financial
discussion, and the reason is not vocabulary or spelling: it is that the two
domains have different distributions of stance, and the model has memorised the
wrong one.

## What this does not show

- **Platform is not controlled.** Post length is matched, but these are forum
  replies and AfriSenti is tweets. Some of the 15.5 points is register rather
  than subject matter, and nothing here separates the two.
- **The set is small and single-annotator.** 109 posts, 18 of them negative.
  Per-class figures on 18 examples are indicative, not precise.
- **The exclusion rate was 15%**, mostly posts that turned out to be standard
  English rather than Pidgin. The collection filter errs towards keeping
  borderline cases, which is the right default but means the annotator does
  that work.
