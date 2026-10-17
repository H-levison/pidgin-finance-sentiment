# Annotation reliability and adjudication

Two annotators, both native Nigerian Pidgin speakers, labelled the finance set independently against `data/annotation/GUIDELINES.md`. Neither saw the other's labels.

## Independent agreement, before adjudication

| judgement | posts | raw agreement | Cohen's kappa | reading |
|---|---|---|---|---|
| what to exclude (not Pidgin, not finance) | 125 | 85.6% | **0.539** | moderate |
| positive against negative, neutral dropped | 33 | 69.7% | **0.368** | fair |
| all three sentiment classes | 92 | 40.2% | **0.088** | slight |

Muhammad et al. (2023b) report a free marginal kappa of 0.6 for Pidgin in AfriSenti. The three-way figure here is well below that, and the decomposition shows why: the disagreement sits almost entirely on the neutral class.

| | annotator 2: negative | neutral | positive |
|---|---|---|---|
| **annotator 1: negative** | 8 | 6 | 4 |
| **annotator 1: neutral** | 10 | 14 | 18 |
| **annotator 1: positive** | 6 | 11 | 15 |

## Adjudication

The 52 posts the annotators already agreed on were left untouched. The 73 disagreements were worked through together, grouped by the kind of disagreement so a decision about a class of case could be applied along a run.

| disagreement | n | settled as |
|---|---|---|
| neutral vs positive | 29 | 14 neutral, 14 positive, 1 exclude |
| negative vs neutral | 16 | 7 neutral, 5 negative, 2 exclude, 2 positive |
| exclude vs neutral | 11 | 8 exclude, 3 neutral |
| negative vs positive | 10 | 3 positive, 3 neutral, 2 exclude, 2 negative |
| exclude vs positive | 6 | 4 exclude, 2 positive |
| exclude vs negative | 1 | 1 negative |

The settled labels match annotator 1 on 73.6% of posts and annotator 2 on 60.0%, so this was not one annotator conceding to the other.

### Were the recorded reasons applied as rules?

| reason recorded | n | maps to one label |
|---|---|---|
| statement | 22 | 50% |
| sentiment | 20 | 50% |
| out of context | 13 | 100% |
| fact | 12 | 92% |
| english | 3 | 100% |
| postive | 1 | 100% |
| not finance related | 1 | 100% |
| comment | 1 | 100% |

The exclusion criteria were applied consistently. Reasons that describe what the annotator saw rather than a rule they applied do not map reliably to a label. The adjudication settled the labels but did not sharpen the guideline; the informational-against-approving boundary needs worked examples defined before labelling, not recovered afterwards.

## The adjudicated gold set

| label | count | share |
|---|---|---|
| negative | 16 | 17.2% |
| neutral | 41 | 44.1% |
| positive | 36 | 38.7% |

Usable test posts: **93**, with 32 excluded.
