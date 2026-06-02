import streamlit as st
import requests
import io
import threading
import time
import re
from urllib.parse import urlparse

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fine-Tune Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────
defaults = {
    "model_ready": False,
    "training_done": False,
    "processing": False,
    "chat_history": [],
    "chunks": [],
    "log_messages": [],
    "trained_model": None,
    "trained_tokenizer": None,
    "doc_text": "",
    "training_progress": 0,
    "source_type": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{ts}] {msg}")


def split_into_chunks(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def extract_text_from_pdf_url(url: str) -> str:
    from pypdf import PdfReader

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_website(url: str) -> str:
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        result = trafilatura.extract(downloaded)
        if result:
            return result
    except Exception:
        pass
    # Fallback: BeautifulSoup (html.parser is built-in, no system deps needed)
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def detect_url_type(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    # Try HEAD to check content-type
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        ct = resp.headers.get("Content-Type", "")
        if "pdf" in ct:
            return "pdf"
    except Exception:
        pass
    return "website"


def run_training_pipeline(url: str, method: str, params: dict):
    """Runs in a background thread."""
    try:
        log("🔍 Detecting URL type...")
        url_type = detect_url_type(url)
        st.session_state.source_type = url_type
        log(f"📄 Source detected: {url_type.upper()}")

        # 1. Extract text
        log("📥 Extracting text from source...")
        if url_type == "pdf":
            text = extract_text_from_pdf_url(url)
        else:
            text = extract_text_from_website(url)

        if not text or len(text.strip()) < 100:
            log("❌ Could not extract meaningful text. Check the URL.")
            st.session_state.processing = False
            return

        st.session_state.doc_text = text
        log(f"✅ Text extracted: {len(text.split())} words")

        # 2. Chunking
        chunk_size = params.get("chunk_size", 500)
        log(f"✂️  Splitting into chunks (size={chunk_size})...")
        chunks = split_into_chunks(text, chunk_size)
        st.session_state.chunks = chunks
        log(f"✅ {len(chunks)} chunks ready")

        # 3. Build Dataset
        log("📦 Building Hugging Face dataset...")
        from datasets import Dataset

        dataset_list = [
            {
                "text": (
                    f"### Instruction:\nUnderstand and remember the following text:\n\n"
                    f"### Input:\n{chunk}\n\n"
                    f"### Response:\nI have learned this information."
                )
            }
            for chunk in chunks
        ]
        hf_dataset = Dataset.from_list(dataset_list)
        log(f"✅ Dataset ready: {len(hf_dataset)} examples")

        # 4. Load Model
        model_name = params.get("model_name", "unsloth/Llama-3.2-1B")
        max_seq_len = params.get("max_seq_length", 2048)
        log(f"🤗 Loading model: {model_name} ...")
        from unsloth import FastLanguageModel

        load_4bit = method == "QLORA"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            load_in_4bit=load_4bit,
        )
        log(f"✅ Model loaded ({method} mode, 4bit={load_4bit})")

        # 5. Apply PEFT
        log(f"⚙️  Applying {method} adapter...")
        use_gc = "unsloth" if method == "QLORA" else True
        model = FastLanguageModel.get_peft_model(
            model,
            r=params.get("r", 16),
            target_modules=["q_proj", "v_proj"],
            lora_alpha=params.get("lora_alpha", 16),
            lora_dropout=params.get("lora_dropout", 0.0),
            bias="none",
            use_gradient_checkpointing=use_gc,
        )
        log(f"✅ {method} adapter applied (r={params['r']}, alpha={params['lora_alpha']})")

        # 6. Train
        log("🏋️  Starting training...")
        from trl import SFTTrainer
        from transformers import TrainingArguments

        output_dir = "qlora_model" if method == "QLORA" else "lora_model"
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_len,
            args=TrainingArguments(
                per_device_train_batch_size=params.get("batch_size", 2),
                gradient_accumulation_steps=params.get("gradient_accumulation_steps", 4),
                warmup_steps=params.get("warmup_steps", 10),
                max_steps=params.get("max_steps", 100),
                learning_rate=params.get("learning_rate", 2e-4),
                output_dir=output_dir,
                logging_steps=10,
                fp16=not load_4bit,
                bf16=False,
            ),
        )
        trainer.train()
        log("✅ Training complete!")

        # 7. Prepare for inference
        FastLanguageModel.for_inference(model)
        st.session_state.trained_model = model
        st.session_state.trained_tokenizer = tokenizer
        st.session_state.training_done = True
        st.session_state.model_ready = True
        log("🚀 Model ready for chat!")

    except Exception as e:
        log(f"❌ Error: {str(e)}")
    finally:
        st.session_state.processing = False


def generate_response(query: str) -> str:
    model = st.session_state.trained_model
    tokenizer = st.session_state.trained_tokenizer
    if model is None or tokenizer is None:
        return "⚠️ Model not loaded yet."

    import torch

    prompt = f"### Instruction:\n{query}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only the response part
    if "### Response:" in decoded:
        return decoded.split("### Response:")[-1].strip()
    return decoded.strip()


# ─────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
/* Dark theme base */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d2f3e; }

/* Header */
.app-header {
    background: linear-gradient(135deg, #6c63ff 0%, #3ecfcf 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.app-header h1 { color: #fff; margin: 0; font-size: 1.8rem; }
.app-header p  { color: rgba(255,255,255,0.85); margin: 4px 0 0; font-size: 0.9rem; }

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.status-ready   { background: #1a3a2a; color: #4ade80; border: 1px solid #4ade80; }
.status-training{ background: #2a2a1a; color: #fbbf24; border: 1px solid #fbbf24; }
.status-idle    { background: #1a1d27; color: #94a3b8; border: 1px solid #2d2f3e; }

/* Chat bubble */
.chat-user {
    background: #6c63ff22;
    border: 1px solid #6c63ff55;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    color: #e2e8f0;
}
.chat-bot {
    background: #1e2435;
    border: 1px solid #2d2f3e;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    color: #e2e8f0;
}
.chat-label {
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Log box */
.log-container {
    background: #0a0c12;
    border: 1px solid #1e2435;
    border-radius: 8px;
    padding: 12px 14px;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #4ade80;
    max-height: 200px;
    overflow-y: auto;
}

/* Disable overlay */
.chat-disabled-overlay {
    background: #0f111788;
    border: 1px dashed #2d2f3e;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    color: #475569;
}

/* Param card */
.param-card {
    background: #1a1d27;
    border: 1px solid #2d2f3e;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}

/* Section title */
.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6c63ff;
    margin-bottom: 10px;
}

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #6c63ff, #3ecfcf); }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="app-header">
    <div style="font-size:2.4rem">🤖</div>
    <div>
        <h1>Fine-Tune Chat</h1>
        <p>Load any URL → Fine-tune with LoRA / QLoRA → Chat with your data</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">📎 Data Source</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "Enter URL",
        placeholder="https://example.com/paper.pdf  or  https://example.com",
        label_visibility="collapsed",
    )

    # Method selector
    st.markdown('<div class="section-title" style="margin-top:18px">⚡ Fine-Tuning Method</div>', unsafe_allow_html=True)
    method = st.radio(
        "Method",
        ["LORA", "QLORA"],
        horizontal=True,
        label_visibility="collapsed",
        help="LoRA = standard, QLoRA = 4-bit quantized (less VRAM)",
    )

    # Parameters
    st.markdown(
        f'<div class="section-title" style="margin-top:18px">🎛️ {method} Parameters</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Model & Data", expanded=True):
        model_name = st.selectbox(
            "HuggingFace Model",
            [
                "unsloth/Llama-3.2-1B",
                "unsloth/Llama-3.2-3B",
                "unsloth/mistral-7b-bnb-4bit",
                "unsloth/phi-2",
                "unsloth/gemma-2b",
            ],
        )
        chunk_size = st.select_slider(
            "Chunk Size (words)", options=[200, 300, 400, 500, 600, 800, 1000], value=500
        )
        max_seq_length = st.select_slider(
            "Max Seq Length", options=[512, 1024, 2048, 4096], value=2048
        )

    with st.expander("LoRA Adapter", expanded=True):
        r = st.select_slider("Rank (r)", options=[4, 8, 16, 32, 64], value=16)
        lora_alpha = st.select_slider("LoRA Alpha", options=[4, 8, 16, 32, 64], value=16)
        lora_dropout = st.slider("LoRA Dropout", 0.0, 0.5, 0.0, 0.05)

    with st.expander("Training", expanded=True):
        batch_size = st.select_slider("Batch Size", options=[1, 2, 4, 8], value=2)
        gradient_accumulation_steps = st.select_slider(
            "Gradient Accum Steps", options=[1, 2, 4, 8, 16], value=4
        )
        max_steps = st.select_slider(
            "Max Steps", options=[10, 20, 50, 100, 200, 500], value=100
        )
        warmup_steps = st.select_slider(
            "Warmup Steps", options=[5, 10, 20, 50], value=10
        )
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            value=2e-4,
            format_func=lambda x: f"{x:.0e}",
        )

    st.markdown("---")
    process_btn = st.button(
        "🚀 Process & Fine-Tune",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.processing,
    )

    # Status
    st.markdown("---")
    if st.session_state.model_ready:
        st.markdown('<span class="status-badge status-ready">✅ Model Ready</span>', unsafe_allow_html=True)
    elif st.session_state.processing:
        st.markdown('<span class="status-badge status-training">⏳ Training...</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">💤 Idle — Paste URL to start</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Process Button Logic
# ─────────────────────────────────────────────
if process_btn:
    if not url_input.strip():
        st.sidebar.error("⚠️ Please enter a URL first.")
    else:
        st.session_state.processing = True
        st.session_state.training_done = False
        st.session_state.model_ready = False
        st.session_state.log_messages = []
        st.session_state.chat_history = []

        params = {
            "model_name": model_name,
            "chunk_size": chunk_size,
            "max_seq_length": max_seq_length,
            "r": r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "learning_rate": learning_rate,
        }

        t = threading.Thread(
            target=run_training_pipeline,
            args=(url_input.strip(), method, params),
            daemon=True,
        )
        t.start()
        st.rerun()

# ─────────────────────────────────────────────
# Main Content Area
# ─────────────────────────────────────────────
col_chat, col_info = st.columns([3, 1], gap="large")

# ── RIGHT COLUMN: Info / Logs ──────────────────
with col_info:
    # Stats card
    if st.session_state.doc_text:
        words = len(st.session_state.doc_text.split())
        n_chunks = len(st.session_state.chunks)
        st.markdown(
            f"""
<div class="param-card">
  <div class="section-title">📊 Document Stats</div>
  <div style="color:#e2e8f0;font-size:0.85rem">
    📝 <b>{words:,}</b> words extracted<br>
    🧩 <b>{n_chunks}</b> chunks created<br>
    🔗 Type: <b>{(st.session_state.source_type or "—").upper()}</b>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Live training log
    if st.session_state.log_messages or st.session_state.processing:
        st.markdown('<div class="section-title">🖥️ Training Log</div>', unsafe_allow_html=True)
        log_text = "\n".join(st.session_state.log_messages[-20:]) or "Waiting..."
        st.markdown(
            f'<div class="log-container">{log_text.replace(chr(10),"<br>")}</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.processing:
            st.spinner("Training in progress...")
            time.sleep(2)
            st.rerun()

    # Parameters summary
    if st.session_state.model_ready:
        st.markdown(
            f"""
<div class="param-card" style="margin-top:12px">
  <div class="section-title">⚙️ Trained Config</div>
  <div style="color:#94a3b8;font-size:0.78rem;line-height:1.7">
    Method: <b style="color:#6c63ff">{method}</b><br>
    Model: <code style="color:#3ecfcf">{model_name.split("/")[-1]}</code><br>
    Rank r: <b>{r}</b> | Alpha: <b>{lora_alpha}</b><br>
    Steps: <b>{max_steps}</b> | LR: <b>{learning_rate:.0e}</b>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

# ── LEFT COLUMN: Chat ─────────────────────────
with col_chat:
    st.markdown('<div class="section-title">💬 Chat</div>', unsafe_allow_html=True)

    # Chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            if not st.session_state.model_ready:
                st.markdown(
                    """
<div class="chat-disabled-overlay">
    <div style="font-size:2rem;margin-bottom:8px">🔒</div>
    <div style="font-size:1rem;color:#64748b">Chat is disabled</div>
    <div style="font-size:0.82rem;margin-top:6px;color:#475569">
        Paste a URL in the sidebar and click <b>Process & Fine-Tune</b> to unlock chat.
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
<div class="chat-disabled-overlay" style="border-color:#6c63ff55">
    <div style="font-size:2rem;margin-bottom:8px">🚀</div>
    <div style="font-size:1rem;color:#6c63ff">Model is ready!</div>
    <div style="font-size:0.82rem;margin-top:6px;color:#94a3b8">
        Ask anything about the document you loaded.
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-label">You</div>'
                        f'<div class="chat-user">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-label">🤖 Model</div>'
                        f'<div class="chat-bot">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

    # Input row (disabled until model ready)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    input_col, btn_col = st.columns([5, 1])

    with input_col:
        user_query = st.text_input(
            "Ask a question",
            placeholder="Ask about the document..." if st.session_state.model_ready else "⚠️ Fine-tune a model first...",
            disabled=not st.session_state.model_ready,
            label_visibility="collapsed",
            key="chat_input",
        )

    with btn_col:
        send_btn = st.button(
            "Send ➤",
            disabled=not st.session_state.model_ready,
            use_container_width=True,
            type="primary",
        )

    if send_btn and user_query.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_query.strip()})
        with st.spinner("Generating response..."):
            response = generate_response(user_query.strip())
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
