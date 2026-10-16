"""Chat interface for the Nigerian Pidgin sentiment classifier.

Loads the fine-tuned model from the Hugging Face Hub, so the same file runs
locally and on Streamlit Community Cloud without change.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "HonourGod001/pidgin-sentiment-afroxlmr"
MAX_LENGTH = 128

st.set_page_config(
    page_title="Pidgin sentiment",
    page_icon=":material/forum:",
    layout="centered",
)

# Markers that are rarely standard English. Used only to warn the viewer when
# the input does not look like Pidgin, never to change the prediction.
PIDGIN_MARKERS = re.compile(
    r"(?<![a-z])(abeg|abi|biko|chop|comot|dem|dey|ehn|gbadun|jare|naim|nawa|"
    r"oga|pikin|sabi|sef|sha|shey|una|waka|wahala|wetin|wey|yawa)(?![a-z])"
)

EXAMPLES = [
    "dis platform na scam, make you no put your money there",
    "i just withdraw my money, e enter sharp sharp",
    "btc price reach hundred thousand dollar today",
    "na very good exchange, dem swallow my money since january",
]


@st.cache_resource(show_spinner=False)
def load_model():
    """Cached across sessions: the model is read-only and slow to load."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def classify(text):
    tokenizer, model = load_model()
    encoded = tokenizer(
        text.lower().strip(), truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1)[0].tolist()
    id2label = model.config.id2label
    return {id2label[i]: float(p) for i, p in enumerate(probabilities)}


def verdict_markdown(scores):
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    label, score = ranked[0]
    icon = {"negative": "🔴", "neutral": "⚪", "positive": "🔵"}[label]
    return f"{icon} **{label}** · {score:.0%} confident", ranked


st.title("Pidgin sentiment")
st.caption(
    "Type Nigerian Pidgin about money, crypto or trading. The model replies "
    "with the sentiment it reads."
)

if "history" not in st.session_state:
    st.session_state.history = []
if "queued" not in st.session_state:
    st.session_state.queued = None

with st.sidebar:
    st.subheader("About this model")
    st.markdown(
        """
Fine-tuned from **AfroXLM-R** on the Nigerian Pidgin portion of **AfriSenti**
(SemEval-2023 Task 12).

Test accuracy **0.697** across three seeds, against **0.676** published for this
model and language.
"""
    )
    st.divider()
    st.subheader("It has three outputs")
    st.markdown(
        """
`negative` · `neutral` · `positive`

That is the whole set. The classifier cannot abstain or say *not applicable* —
a softmax spreads every input across these three, so unrelated text still gets
a label.
"""
    )
    st.divider()
    st.subheader("Neutral never wins")
    st.markdown(
        """
The training split held **51** neutral examples against roughly **3,400**
negative and positive ones. On the held-out test set the model scored an F1 of
**0.000** on neutral across all three seeds, never once predicting it.

Watch the neutral bar: it sits near zero whatever you type.
"""
    )
    if st.button("Clear conversation", icon=":material/restart_alt:"):
        st.session_state.history = []
        st.rerun()

if not st.session_state.history:
    st.markdown("**Try one of these**")
    columns = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        with columns[i % 2]:
            if st.button(example, key=f"ex{i}", width="stretch"):
                st.session_state.queued = example
                st.rerun()

for turn in st.session_state.history:
    with st.chat_message("user", avatar=":material/person:"):
        st.write(turn["text"])
    with st.chat_message("assistant", avatar=":material/analytics:"):
        st.markdown(turn["verdict"])
        frame = pd.DataFrame(
            {"label": [k for k, _ in turn["ranked"]],
             "probability": [v for _, v in turn["ranked"]]}
        )
        st.bar_chart(frame, x="label", y="probability", horizontal=True, height=150)
        for note in turn["notes"]:
            st.caption(note)

typed = st.chat_input("Write Pidgin here…")
text = typed or st.session_state.queued
st.session_state.queued = None

if text and text.strip():
    with st.spinner("Reading…"):
        scores = classify(text)
    verdict, ranked = verdict_markdown(scores)

    notes = []
    if not PIDGIN_MARKERS.search(text.lower()):
        notes.append(
            "No obvious Pidgin markers in this text. The model was trained on "
            "Pidgin only, so treat the result as unreliable."
        )
    if ranked[0][1] < 0.70:
        notes.append(
            "Low confidence. These probabilities are uncalibrated and do not "
            "separate the model's right answers from its wrong ones."
        )
    if scores.get("neutral", 0) < 0.10:
        notes.append(
            f"Neutral scored {scores.get('neutral', 0):.0%}. It is pinned near "
            "zero for almost any input, which is this project's main finding."
        )

    st.session_state.history.append(
        {"text": text, "verdict": verdict, "ranked": ranked, "notes": notes}
    )
    st.rerun()

st.divider()
st.caption(
    "Not financial advice. University NLP project — "
    "[model card](https://huggingface.co/HonourGod001/pidgin-sentiment-afroxlmr)"
)
