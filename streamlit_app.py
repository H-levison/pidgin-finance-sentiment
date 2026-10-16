"""Streamlit interface for the Nigerian Pidgin sentiment classifier.

Loads the fine-tuned model from the Hugging Face Hub rather than from a local
directory, so the same file runs locally and on Streamlit Community Cloud
without any change.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "HonourGod001/pidgin-sentiment-afroxlmr"
MAX_LENGTH = 128

st.set_page_config(
    page_title="Pidgin sentiment",
    page_icon=":material/sentiment_satisfied:",
    layout="centered",
)

# Fast UI first: the title and inputs render before the model finishes loading.
st.title("Nigerian Pidgin sentiment")
st.caption(
    "Fine-tuned from AfroXLM-R on the Nigerian Pidgin portion of AfriSenti "
    "(SemEval-2023 Task 12), and tested additionally on hand-labelled Pidgin "
    "posts about money, crypto and peer-to-peer trading."
)


@st.cache_resource(show_spinner="Loading the model, this takes a moment the first time")
def load_model():
    """Cached across reruns and sessions: loading is slow and the model is
    read-only, so every visitor shares one copy."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def classify(text):
    tokenizer, model = load_model()
    encoded = tokenizer(
        text.lower().strip(),
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1)[0].tolist()
    # Read the label order from the model config rather than assuming it.
    id2label = model.config.id2label
    return {id2label[i]: float(p) for i, p in enumerate(probabilities)}


EXAMPLES = {
    "Scam warning": "dis platform na scam, make you no put your money there",
    "Happy withdrawal": "i just withdraw my money, e enter sharp sharp. dem no dey delay",
    "Plain price report": "btc price reach hundred thousand dollar today",
    "Question, no stance": "abeg how i go withdraw from p2p?",
    "Sarcasm": "na very good exchange, dem swallow my money since january",
    "Frustration": "i dey tire for this naira wahala, everything just dey cost",
}

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES["Scam warning"]

with st.container(border=True):
    st.subheader("Try an example", divider=False)
    st.caption(
        "The last four are chosen to show where the model fails, not only where "
        "it works."
    )
    choice = st.pills(
        "Examples",
        options=list(EXAMPLES),
        label_visibility="collapsed",
    )
    if choice:
        st.session_state.text = EXAMPLES[choice]

text = st.text_area(
    "Pidgin text",
    key="text",
    height=110,
    placeholder="Type or paste Nigerian Pidgin here",
)

if text and text.strip():
    scores = classify(text)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_score = ranked[0]

    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted", top_label)
    with right:
        st.metric("Confidence", f"{top_score:.0%}")

    frame = pd.DataFrame(
        {"label": [k for k, _ in ranked], "probability": [v for _, v in ranked]}
    )
    st.bar_chart(frame, x="label", y="probability", horizontal=True, height=180)

    if top_label != "neutral" and top_score < 0.75:
        st.info(
            "Low confidence. The model's probabilities do not separate its "
            "right answers from its wrong ones, so treat this as a guess.",
            icon=":material/info:",
        )

st.divider()

with st.expander("Known limitation: the model never predicts neutral"):
    st.markdown(
        """
The training split contains **51 neutral examples** against roughly 3,400
negative and positive ones. On the held-out test set the model scores an F1 of
**exactly 0.000 on neutral across three random seeds**, having never once
predicted the class for any of its 431 neutral examples.

This is not a tuning problem. Weighting the neutral class 22x in the loss
function did not change it: reweighting moves the decision boundary but adds no
information about what neutral looks like.

Try *"btc price reach hundred thousand dollar today"* — a plain statement of
fact with no stance, which the model is forced to call positive or negative.
"""
    )

with st.expander("Model, data and how to read the numbers"):
    st.markdown(
        """
**Model** — `Davlan/afro-xlmr-base` (XLM-R adapted to 17 African languages
including Nigerian Pidgin) fine-tuned for three-way sentiment classification.

**Data** — AfriSenti, the Nigerian Pidgin (pcm) split, released under
CC BY-NC-SA 4.0 and used here for non-commercial coursework.

**Test results** (3 seeds): accuracy 0.697, macro F1 0.479, weighted F1 0.658.
For comparison, Muhammad et al. (2023) report 67.6 accuracy for this model on
this split, averaged over 5 runs. This version was trained on 3,486 rows rather
than the released 5,122, because an audit found 1,013 training tweets appearing
verbatim in the test set.

**Confidence is not reliability.** The scores above come from a softmax and are
not calibrated. The model is frequently confident and wrong.
"""
    )

st.caption("Not financial advice. Built as a university NLP project.")
