"""Chat interface for the Nigerian Pidgin sentiment classifier.

Loads the fine-tuned model from the Hugging Face Hub, so the same file runs
locally and on Streamlit Community Cloud without change.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import re

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "HonourGod001/pidgin-sentiment-afroxlmr"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
MAX_LENGTH = 128

st.set_page_config(
    page_title="Pidgin sentiment",
    page_icon=":material/forum:",
    layout="centered",
)

# Pidgin detection, used only to warn the viewer when the input does not look
# like Pidgin. It never changes the prediction.
#
# Two tiers, because a single list fails either way. Words like "dey" and
# "wetin" are almost never standard English, so one is enough. Words like "na",
# "dis" and "make" are ordinary English too, so several are needed before they
# mean anything. An earlier version used only the first tier and reported "no
# Pidgin markers" on plainly Pidgin sentences such as
# "dis platform na scam, make you no put your money there".
STRONG_MARKERS = re.compile(
    r"(?<![a-z])(abeg|abi|biko|chop|comot|dem|dey|ehn|gbadun|jare|naim|nawa|"
    r"oga|pikin|sabi|sef|sha|shey|una|waka|wahala|wetin|wey|yawa|oyibo|"
    r"gist|jor|kpata|wahalla)(?![a-z])"
)
WEAK_MARKERS = re.compile(
    r"(?<![a-z])(na|don|fit|nor|dat|dis|im|go|make|say|no|get|take|put|"
    r"plenty|small|well|dey|una|e)(?![a-z])"
)


def looks_like_pidgin(text, min_weak=1):
    """True if there is any sign of Pidgin.

    The warning this feeds should be rare, so the bar is deliberately low: one
    marker of either tier is enough. A heuristic this crude will never classify
    Pidgin reliably, and a warning that fires on real Pidgin is worse than one
    that occasionally stays quiet.
    """
    lowered = text.lower()
    if STRONG_MARKERS.search(lowered):
        return True
    return len(set(WEAK_MARKERS.findall(lowered))) >= min_weak


# Sentiment is a polarity scale, so the two poles get warm and cool hues and the
# midpoint stays grey. Streamlit's own colour names keep this readable in both
# light and dark mode without the app having to know which is active.
TONE = {"negative": "red", "neutral": "gray", "positive": "blue"}

EXAMPLES = [
    ("Scam warning", "dis platform na scam, make you no put your money there"),
    ("Fast payout", "i just withdraw my money, e enter sharp sharp"),
    ("Plain fact", "btc price reach hundred thousand dollar today"),
    ("Sarcasm", "na very good exchange, dem swallow my money since january"),
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


def notes_for(text, scores, top_label, top_score):
    """Flag the cases where this answer should not be trusted.

    Only conditions that are genuinely exceptional belong here. The neutral
    score sits near zero for any input, so warning about it on every reply would be
    constant noise; it is stated once in the sidebar instead.
    """
    notes = []
    if not looks_like_pidgin(text):
        notes.append(
            "This may not be Pidgin. The model was trained on Pidgin only, "
            "so the answer is unreliable here."
        )
    if top_score < 0.70:
        notes.append("Low confidence, and these scores are not calibrated.")
    return notes


def render_reply(turn):
    scores, top_label, top_score = turn["scores"], turn["label"], turn["score"]
    st.markdown(f"### :{TONE[top_label]}[{top_label}]  ·  {top_score:.0%}")
    for label, value in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        left, right = st.columns([5, 1], vertical_alignment="center")
        with left:
            st.progress(value, text=f":{TONE[label]}[{label}]")
        with right:
            st.markdown(
                f"<div style='text-align:right;font-variant-numeric:tabular-nums;"
                f"opacity:.75'>{value:.0%}</div>",
                unsafe_allow_html=True,
            )
    for note in turn["notes"]:
        st.caption(note)


with st.sidebar:
    st.markdown("#### Pidgin sentiment")
    st.caption(
        f"AfroXLM-R fine-tuned on AfriSenti (pcm). Test accuracy **0.697**, "
        f"against 0.676 published. [Model card]({MODEL_URL})"
    )
    with st.expander("Why it never says neutral"):
        st.markdown(
            """
It has exactly three outputs — negative, neutral, positive — and no way to
abstain, so a softmax spreads every input across them. Unrelated text still
gets a label.

Neutral is the broken one. The training split held **51** neutral examples
against roughly 3,400 of the others, and this model scored F1 **0.000** on
neutral across three seeds, never predicting it once. That is the model
running here.

A separate experiment added every other labelled neutral Pidgin corpus that
exists, roughly doubling the class, and lifted it only to 0.035 — the first
neutral predictions, but still 1.9% recall. That variant is reported in the
results, not deployed here.
"""
        )
    if st.session_state.get("history"):
        if st.button("Clear", icon=":material/restart_alt:", width="stretch"):
            st.session_state.history = []
            st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []
if "queued" not in st.session_state:
    st.session_state.queued = None

if not st.session_state.history:
    st.title("Pidgin sentiment")
    st.caption(
        "Write Nigerian Pidgin about money, crypto or trading. "
        "The last two examples are ones it gets wrong."
    )
    columns = st.columns(2)
    for i, (name, example) in enumerate(EXAMPLES):
        with columns[i % 2]:
            if st.button(
                f"**{name}**  \n{example[:38]}…",
                key=f"ex{i}",
                width="stretch",
            ):
                st.session_state.queued = example
                st.rerun()

for turn in st.session_state.history:
    with st.chat_message("user", avatar=":material/person:"):
        st.write(turn["text"])
    with st.chat_message("assistant", avatar=":material/graphic_eq:"):
        render_reply(turn)

typed = st.chat_input("Write Pidgin here…")
text = typed or st.session_state.queued
st.session_state.queued = None

if text and text.strip():
    with st.spinner("Reading…"):
        scores = classify(text)
    top_label, top_score = max(scores.items(), key=lambda kv: kv[1])
    st.session_state.history.append({
        "text": text,
        "scores": scores,
        "label": top_label,
        "score": top_score,
        "notes": notes_for(text, scores, top_label, top_score),
    })
    st.rerun()

if st.session_state.history:
    st.caption("Not financial advice. University NLP project.")
