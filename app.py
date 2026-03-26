import os
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

st.set_page_config(
    page_title="BART Text Summarizer",
    page_icon="📝",
    layout="wide",
)

DEFAULT_MODEL_DIR = os.getenv("MODEL_DIR", "./model")
REQUIRED_MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


def model_dir_is_ready(model_dir: str) -> tuple[bool, list[str]]:
    path = Path(model_dir)
    missing = [name for name in REQUIRED_MODEL_FILES if not (path / name).exists()]
    return len(missing) == 0, missing


@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = BartForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def summarize_text(
    text: str,
    tokenizer,
    model,
    device,
    max_length: int,
    min_length: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    length_penalty: float,
):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary


st.title("Phase 4 BART Summarization App")
st.caption("Production style interface for abstractive text summarization with the fine tuned BART model.")

with st.sidebar:
    st.header("Deployment settings")
    st.write("This app loads a local fine tuned BART model and serves inference through Streamlit.")
    model_dir = st.text_input("Model directory", value=DEFAULT_MODEL_DIR)

    max_length = st.slider("Max summary length", min_value=30, max_value=180, value=60, step=5)
    min_length = st.slider("Min summary length", min_value=10, max_value=100, value=20, step=5)
    num_beams = st.slider("Beam size", min_value=1, max_value=8, value=4, step=1)
    no_repeat_ngram_size = st.slider("No repeat ngram size", min_value=1, max_value=6, value=3, step=1)
    length_penalty = st.slider("Length penalty", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

    st.divider()
    st.subheader("Architecture")
    st.write(
        "The deployed model is a fine tuned BART checkpoint for abstractive summarization. "
        "The app reuses the trained model assets from the earlier phases and exposes inference through a simple UI."
    )

ready, missing_files = model_dir_is_ready(model_dir)
if not ready:
    st.error(
        "The model directory is not ready. Place your final model files inside the selected folder.\n\n"
        f"Missing files: {', '.join(missing_files)}"
    )
    st.stop()

try:
    tokenizer, model, device = load_model_and_tokenizer(model_dir)
except Exception as exc:
    st.error(f"The model could not be loaded from '{model_dir}'. Details: {exc}")
    st.stop()

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Input article")
    user_text = st.text_area(
        "Paste the article or long text to summarize",
        height=320,
        placeholder="Enter the article text here...",
    )
with col2:
    st.subheader("Runtime")
    st.write(f"Device: {device}")
    st.write(f"Model path: {Path(model_dir).resolve()}")
    st.write(f"Input words: {len(user_text.split()) if user_text else 0}")

sample_text = (
    "The BART model is a sequence to sequence transformer that can generate abstractive summaries. "
    "In this project, the model was fine tuned on a news summarization task and is now deployed through a "
    "Streamlit interface so users can paste an article and receive a concise summary."
)

if st.button("Load sample text"):
    st.session_state["sample_text_loaded"] = sample_text

if st.session_state.get("sample_text_loaded") and not user_text:
    user_text = sample_text

summarize_clicked = st.button("Generate summary", type="primary", use_container_width=True)

if summarize_clicked:
    cleaned_text = user_text.strip()
    if not cleaned_text:
        st.warning("Please enter some text before generating a summary.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_text(
                text=cleaned_text,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
            )

        st.subheader("Generated summary")
        st.success(summary)

        st.subheader("Quick comparison")
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric("Input word count", len(cleaned_text.split()))
        with result_col2:
            st.metric("Summary word count", len(summary.split()))

with st.expander("How to use this app"):
    st.write(
        "1. Make sure the final model files are inside the local model folder. 2. Paste a news article or any long text. "
        "3. Adjust the decoding parameters if needed. 4. Generate the summary and compare the output length with the input."
    )
