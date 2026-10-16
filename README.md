# Sentiment classification for Nigerian Pidgin, with a finance and crypto test

Sentiment analysis for Nigerian Pidgin (ISO code `pcm`), trained on AfriSenti
and then tested on a second, hand-labelled set of Pidgin posts about money,
crypto and peer-to-peer trading. The question the project is built around is not
just "can a model classify Pidgin sentiment" but "does a model trained on
general Pidgin tweets still work when people are talking about money".

**Links**

- Fine-tuned model: https://huggingface.co/HonourGod001/pidgin-sentiment-afroxlmr
- Live app: _Streamlit Community Cloud, link to add after deploy_
- Demo video: _to add_
- Report: _to add_

The app is a Streamlit interface (`streamlit_app.py`) that loads the model from
the Hugging Face Hub, so the same file runs locally and in the cloud:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The fine-tuned model is hosted on the Hugging Face Hub and the app loads it from
there, so no checkpoint is committed to this repository.

## Why this task and this language

Nigerian Pidgin is spoken by a very large number of people across West Africa
and is heavily used on Nigerian social media, but it has far fewer NLP resources
than its speaker count suggests. It is also the language a lot of Nigerian
crypto and fintech conversation actually happens in, which matters if anyone
wants to track how users talk about a payment service, an exchange or a scam.

There are published sentiment models for Pidgin. What does not exist, as far as
I could find, is any labelled Pidgin data for the finance and crypto domain, so
nobody has measured whether those models transfer to it. That gap is what the
second test set in this project is for.

## Dataset

Primary data is the Nigerian Pidgin portion of **AfriSenti** (SemEval-2023
Task 12), downloaded from the official task repository. Three sentiment classes:
negative, neutral, positive. Licence is CC BY-NC-SA 4.0, so it is used here for
non-commercial coursework only, and the raw files are not committed to this
repository. Run `python src/download_data.py` to fetch them.

### What the audit found

Running `src/prepare_data.py` surfaced two problems in the release that shaped
the whole project.

**1. The neutral class barely exists in training but is 10% of test.**

| | train | dev | test |
|---|---|---|---|
| negative | 3,241 | 813 | 2,326 |
| positive | 1,808 | 447 | 1,397 |
| neutral | 72 | 21 | 431 |
| neutral share | 1.4% | 1.6% | **10.4%** |

So the test set is drawn from a noticeably different label distribution than
training. Any model is going to struggle on neutral, and not because Pidgin is
hard: because it has almost no examples to learn from.

**2. There is substantial train/test leakage, and the gold labels contradict
each other.**

| check | count |
|---|---|
| duplicate tweets inside train | 512 |
| train tweets appearing verbatim in test | 1,013 |
| train tweets appearing verbatim in dev | 291 |
| dev tweets appearing verbatim in test | 266 |
| share of test reachable verbatim from train | 22.3% |
| label agreement on the train/test overlap | 87.8% |

That last row is the awkward one. On the 1,013 tweets shared between train and
test, the two splits disagree about the correct label 124 times. Those tweets
cannot all be scored correctly by any model, so there is a hard ceiling on the
achievable accuracy that has nothing to do with modelling.

Because of this, `prepare_data.py` writes **two** training files:

- `train_full.csv` — the release as distributed, whitespace-normalised only.
  Results on this stay comparable to published AfriSenti numbers, which were
  computed on these splits.
- `train_dedup.csv` — duplicates removed, and any tweet also appearing in dev or
  test dropped. 3,486 rows instead of 5,121. Harder, but the score means
  something.

Dev and test are never modified, so results remain comparable to prior work.

### Preprocessing

Deliberately light. AfriSenti is already lowercased, stripped of punctuation and
anonymised (mentions replaced with `@user`, URLs removed), so the only cleaning
applied is collapsing whitespace and dropping rows that end up empty. Emoji are
kept: 275 training tweets contain them and they carry sentiment.

## Models

| | why it is in the comparison |
|---|---|
| TF-IDF + logistic regression | baseline, trains in seconds, no pretraining |
| `roberta-base` | English only — Pidgin borrows most of its vocabulary from English, so how far does that alone get you |
| `xlm-roberta-base` | 100 languages, but not Pidgin — does general multilingual pretraining help |
| `Davlan/afro-xlmr-base` | XLM-R adapted to 17 African languages including `pcm` |
| `castorini/afriberta_base` | trained only on African languages, much smaller corpus |

`Davlan/naija-twitter-sentiment-afriberta-large` is deliberately excluded. It is
already fine-tuned on NaijaSenti, which overlaps AfriSenti Pidgin, so evaluating
it here would be testing it on its own training data.

## Results

### Final model, test set, 3 seeds

AfroXLM-R, class-weighted, trained on the deduplicated split, 6 epochs, best
epoch chosen on dev macro F1. The test split was scored once, after the dev
comparison had picked the configuration.

| metric | mean | range |
|---|---|---|
| accuracy | **0.697** | 0.694 - 0.700 |
| macro F1 | 0.479 | 0.475 - 0.483 |
| weighted F1 | 0.658 | 0.654 - 0.663 |
| F1 negative | 0.784 | 0.781 - 0.785 |
| **F1 neutral** | **0.000** | **0.000 - 0.000** |
| F1 positive | 0.653 | 0.644 - 0.665 |

Muhammad et al. (2023b) report 67.6 accuracy for AfroXLMR-base on this split,
averaged over 5 runs. This project reaches 69.7 averaged over 3 runs, trained on
3,486 rows rather than the released 5,122 - the deduplicated set, with every
tweet that also appears in dev or test removed. Removing a third of the training
data, specifically the leaked third, cost nothing measurable.

The neutral row is the main finding: F1 of exactly zero on all three seeds,
against 431 neutral examples in the test set. The model never predicts the class
at all.

### Model comparison (dev set, 6-epoch budget)

| model | pretraining | accuracy | macro F1 | weighted F1 | F1 neutral |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | none | 0.665 | 0.424 | 0.657 | 0.000 |
| Word2Vec + BiLSTM | none | 0.671 | 0.422 | 0.659 | 0.000 |
| AfriBERTa | African only, <1 GB news | 0.687 | 0.437 | 0.677 | 0.000 |
| XLM-R | 100 languages, no pcm | 0.712 | 0.463 | 0.707 | 0.000 |
| RoBERTa | English only | 0.728 | 0.467 | 0.718 | 0.000 |
| **AfroXLM-R** | 17 African languages incl. pcm | **0.743** | 0.471 | 0.727 | 0.000 |

The ordering AfriBERTa < XLM-R < AfroXLMR reproduces the published ranking for
this language (their Table 7: 62.3 / 63.8 / 67.6). Absolute values differ
because these are dev numbers and theirs are test.

Three things worth drawing out:

- **English-only pretraining beats general multilingual pretraining.** RoBERTa
  has never seen an African language and still beats XLM-R, which covers 100
  languages but not Pidgin. Nigerian Pidgin is an English-lexifier creole, so
  most of its vocabulary is English-derived and an English subword vocabulary
  segments it well. XLM-R spreads its capacity across languages that do not help
  here.
- **The neural sequence model does not beat bag-of-words.** The BiLSTM and
  TF-IDF are level (0.671 against 0.665 accuracy), inside the noise floor. With
  3,486 training posts there is not enough data for the extra capacity.
- **Neutral F1 is 0.000 for every honestly trained model**, from logistic
  regression to a 278M-parameter transformer.

### What class weighting does

| model | effect of inverse-frequency weighting |
|---|---|
| TF-IDF | +5.6 macro F1 (large help) |
| BiLSTM | -2.8 accuracy (slight harm) |
| AfroXLM-R | no reliable effect (sign flips between budgets) |

Weighting moves the decision boundary but adds no information. TF-IDF was
collapsing onto the majority class, so the shift rescued positive recall (0.22
to 0.51). The other two were not collapsing that way, so the same shift only
cost them accuracy. Nothing rescues neutral, because 51 examples is an
information problem.

### How much the benchmark's leakage inflates scores

Same model, same budget, three training sets:

| training set | rows | leakage | accuracy | macro F1 | F1 neutral |
|---|---|---|---|---|---|
| `train_dedup` | 3,486 | none | 0.743 | 0.472 | 0.000 |
| `train_full_sub` | 3,486 | retained | 0.748 | 0.511 | 0.077 |
| `train_full` | 5,121 | retained | 0.756 | 0.521 | 0.083 |

`train_full_sub` is a size-matched control: the same row count as the clean set,
with the duplicates and overlaps left in. Comparing it to `train_dedup` isolates
the leakage from the extra data volume that `train_full` also brings.

**Leakage alone is worth about 4 macro F1 points**; the additional 1,635 rows
add roughly 1 more. The neutral class only becomes non-zero in the leaky
conditions, which is the clearest sign of what is happening: the model is
recognising memorised tweets, not learning the class.

### Measurement noise

| source | size (macro F1) |
|---|---|
| seed to seed | 0.008 |
| epoch budget / LR schedule | 0.016 |

Differences smaller than these are not findings. The learning-rate schedule is
computed over total training steps, so changing the epoch budget shifts the
whole schedule and moves results about twice as much as reseeding does.

### Error analysis (test set)

- Neutral does not fail randomly: 61% of neutral posts are called negative and
  39% positive. There is no neutral region in the model at all.
- Short posts are twice as hard: 40.5% error at 1-10 words against 20.6% at 36+.
- Unseen vocabulary barely matters (31% / 27% / 33% across coverage thirds),
  which rules out the obvious explanation that Pidgin spelling variation drives
  the errors.
- Posts containing emoji are *easier* (24.4% error against 31.0%).
- 24 test posts were marked wrong where the model predicted the label the
  training split gives the same tweet. The benchmark contradicts itself there.

## Repository layout

```
src/download_data.py   fetch the AfriSenti pcm splits
src/prepare_data.py    audit, clean, write CSVs and the stats report
src/config.py          paths, label order, seed
src/evaluation.py      scoring, confusion matrices, experiment log
src/baseline.py        TF-IDF + logistic regression
src/finetune.py        transformer fine-tuning
results/               metrics, figures, experiments.csv
app/                   web interface
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-train.txt

python src/download_data.py
python src/prepare_data.py
python src/baseline.py --class-weight balanced
python src/finetune.py --model Davlan/afro-xlmr-base --class-weight
```

Checkpoints are written to `./models` by default. Set `PIDGIN_MODEL_DIR` to put
them somewhere else, which is worth doing if the project folder is inside a
cloud-synced directory.

Trained on a single RTX 3060 Laptop GPU (6 GB), using mixed precision to fit the
base-size models.

## Credits

- **AfriSenti** — Muhammad et al. (2023), SemEval-2023 Task 12. Dataset,
  CC BY-NC-SA 4.0.
- **AfroXLM-R** — Alabi et al. (2022), `Davlan/afro-xlmr-base`.
- **AfriBERTa** — Ogueji et al. (2021), `castorini/afriberta_base`.
- **XLM-R** — Conneau et al. (2020).
- Libraries: PyTorch, Hugging Face Transformers, scikit-learn, pandas, Gradio.

Full citations are in the report. Everything in `src/` was written for this
project; the datasets and pretrained models above are existing resources.
