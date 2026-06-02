import streamlit as st
import requests
import io
import subprocess
import sys
import threading
import time
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
    "installing": False,
    "chat_history": [],
    "chunks": [],
    "log_messages": [],
    "trained_model": None,
    "trained_tokenizer": None,
    "doc_text": "",
    "source_type": None,
    "deps_installed": False,
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


def install_ml_deps():
    """Install torch + unsloth + bitsandbytes at runtime (Python-version-aware)."""
    log("📦 Installing torch (this takes ~2 min on first run)...")
    cmds = [
        [sys.executable, "-m", "pip", "install", "--quiet",
         "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
        [sys.executable, "-m", "pip", "install", "--quiet",
         "bitsandbytes>=0.43.1"],
        [sys.executable, "-m", "pip", "install", "--quiet",
         "unsloth", "--no-deps"],
    ]
    for cmd in cmds:
        pkg = cmd[5] if "--index-url" not in cmd else "torch"
        log(f"  ⬇️  Installing {pkg}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"  ⚠️  Warning installing {pkg}: {result.stderr[-200:]}")
        else:
            log(f"  ✅ {pkg} installed")
    st.session_state.deps_installed = True
    log("✅ All ML dependencies ready!")


def split_into_chunks(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]


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
    from bs4 import BeautifulSoup
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def detect_url_type(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return "pdf"
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        if "pdf" in resp.headers.get("Content-Type", ""):
            return "pdf"
    except Exception:
        pass
    return "website"


def run_training_pipeline(url: str, method: str, params: dict):
    """Runs in a background thread."""
    try:
        # Step 0: Install ML deps if needed
        if not st.session_state.deps_installed:
            install_ml_deps()

        # Step 1: Extract text
        log("🔍 Detecting URL type...")
        url_type = detect_url_type(url)
        st.session_state.source_type = url_type
        log(f"📄 Source: {url_type.upper()}")

        log("📥 Extracting text...")
        text = extract_text_from_pdf_url(url) if url_type == "pdf" else extract_text_from_website(url)

        if not text or len(text.strip()) < 100:
            log("❌ Could not extract meaningful text. Check the URL.")
            st.session_state.processing = False
            return

        st.session_state.doc_text = text
        log(f"✅ Extracted {len(text.split()):,} words")

        # Step 2: Chunking
        chunk_size = params.get("chunk_size", 500)
        log(f"✂️  Chunking (size={chunk_size})...")
        chunks = split_into_chunks(text, chunk_size)
        st.session_state.chunks = chunks
        log(f"✅ {len(chunks)} chunks ready")

        # Step 3: Dataset
        log("📦 Building dataset...")
        from datasets import Dataset
        dataset_list = [
            {
                "text": (
                    f"### Instruction:\nUnderstand and remember the following:\n\n"
                    f"### Input:\n{chunk}\n\n"
                    f"### Response:\nI have learned this information."
                )
            }
            for chunk in chunks
        ]
        hf_dataset = Dataset.from_list(dataset_list)
        log(f"✅ Dataset: {len(hf_dataset)} examples")

        # Step 4: Load model
        model_name = params.get("model_name", "unsloth/Llama-3.2-1B")
        max_seq_len = params.get("max_seq_length", 2048)
        load_4bit = (method == "QLORA")
        log(f"🤗 Loading {model_name} ({'4-bit QLoRA' if load_4bit else 'LoRA'})...")

        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            load_in_4bit=load_4bit,
        )
        log("✅ Model loaded")

        # Step 5: Apply adapter
        log(f"⚙️  Applying {method} adapter (r={params['r']}, alpha={params['lora_alpha']})...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=params.get("r", 16),
            target_modules=["q_proj", "v_proj"],
            lora_alpha=params.get("lora_alpha", 16),
            lora_dropout=params.get("lora_dropout", 0.0),
            bias="none",
            use_gradient_checkpointing="unsloth" if load_4bit else True,
        )
        log(f"✅ {method} adapter applied")

        # Step 6: Train
        log("🏋️  Training started...")
        from trl import SFTTrainer
        from transformers import TrainingArguments
        import torch

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
                output_dir="qlora_model" if load_4bit else "lora_model",
                logging_steps=10,
                fp16=not load_4bit,
                bf16=False,
            ),
        )
        trainer.train()
        log("✅ Training complete!")

        # Step 7: Ready for inference
        FastLanguageModel.for_inference(model)
        st.session_state.trained_model = model
        st.session_state.trained_tokenizer = tokenizer
        st.session_state.training_done = True
        st.session_state.model_ready = True
        log("🚀 Model ready — start chatting!")

    except Exception as e:
        log(f"❌ Error: {str(e)}")
    finally:
        st.session_state.processing = False


def generate_response(query: str) -> str:
    model = st.session_state.trained_model
    tokenizer = st.session_state.trained_tokenizer
    if not model or not tokenizer:
        return "⚠️ Model not loaded."
    import torch
    prompt = f"### Instruction:\n{query}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip() if "### Response:" in decoded else decoded.strip()


# ─────────────────────────────────────────────
# CSS Styles
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d2f3e; }

.app-header {
    background: linear-gradient(135deg, #6c63ff 0%, #3ecfcf 100%);
    border-radius: 12px; padding: 20px 28px; margin-bottom: 24px;
}
.app-header h1 { color: #fff; margin: 0; font-size: 1.8rem; }
.app-header p  { color: rgba(255,255,255,0.85); margin: 4px 0 0; font-size: 0.9rem; }

.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;
}
.status-ready    { background:#1a3a2a; color:#4ade80; border:1px solid #4ade80; }
.status-training { background:#2a2a1a; color:#fbbf24; border:1px solid #fbbf24; }
.status-idle     { background:#1a1d27; color:#94a3b8; border:1px solid #2d2f3e; }

.chat-user {
    background:#6c63ff22; border:1px solid #6c63ff55;
    border-radius:16px 16px 4px 16px; padding:12px 16px;
    margin:8px 0; max-width:80%; margin-left:auto; color:#e2e8f0;
}
.chat-bot {
    background:#1e2435; border:1px solid #2d2f3e;
    border-radius:16px 16px 16px 4px; padding:12px 16px;
    margin:8px 0; max-width:80%; color:#e2e8f0;
}
.chat-label { font-size:0.72rem; color:#64748b; margin-bottom:4px;
    text-transform:uppercase; letter-spacing:0.05em; }

.log-container {
    background:#0a0c12; border:1px solid #1e2435; border-radius:8px;
    padding:12px 14px; font-family:'Courier New',monospace; font-size:0.75rem;
    color:#4ade80; max-height:220px; overflow-y:auto; white-space:pre-wrap;
}
.chat-disabled-overlay {
    background:#0f111788; border:1px dashed #2d2f3e; border-radius:12px;
    padding:32px; text-align:center; color:#475569;
}
.section-title {
    font-size:0.75rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.08em; color:#6c63ff; margin-bottom:10px;
}
.param-card {
    background:#1a1d27; border:1px solid #2d2f3e;
    border-radius:10px; padding:14px; margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div style="font-size:2.2rem;display:inline-block;margin-right:12px">🤖</div>
    <div style="display:inline-block;vertical-align:top">
        <h1>Fine-Tune Chat</h1>
        <p>Load any URL → Fine-tune with LoRA / QLoRA → Chat with your data</p>
    </div>
</div>
""", unsafe_allow_html=True)

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

    st.markdown('<div class="section-title" style="margin-top:18px">⚡ Fine-Tuning Method</div>', unsafe_allow_html=True)
    method = st.radio("Method", ["LORA", "QLORA"], horizontal=True, label_visibility="collapsed")

    st.markdown(f'<div class="section-title" style="margin-top:18px">🎛️ {method} Parameters</div>', unsafe_allow_html=True)

    with st.expander("Model & Data", expanded=True):
        model_name = st.selectbox("HuggingFace Model", [
            "unsloth/Llama-3.2-1B",
            "unsloth/Llama-3.2-3B",
            "unsloth/mistral-7b-bnb-4bit",
            "unsloth/phi-2",
            "unsloth/gemma-2b",
        ])
        chunk_size = st.select_slider("Chunk Size (words)", options=[200, 300, 400, 500, 600, 800, 1000], value=500)
        max_seq_length = st.select_slider("Max Seq Length", options=[512, 1024, 2048, 4096], value=2048)

    with st.expander("LoRA Adapter", expanded=True):
        r = st.select_slider("Rank (r)", options=[4, 8, 16, 32, 64], value=16)
        lora_alpha = st.select_slider("LoRA Alpha", options=[4, 8, 16, 32, 64], value=16)
        lora_dropout = st.slider("LoRA Dropout", 0.0, 0.5, 0.0, 0.05)

    with st.expander("Training", expanded=True):
        batch_size = st.select_slider("Batch Size", options=[1, 2, 4, 8], value=2)
        gradient_accumulation_steps = st.select_slider("Gradient Accum Steps", options=[1, 2, 4, 8, 16], value=4)
        max_steps = st.select_slider("Max Steps", options=[10, 20, 50, 100, 200, 500], value=100)
        warmup_steps = st.select_slider("Warmup Steps", options=[5, 10, 20, 50], value=10)
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

    st.markdown("---")
    if st.session_state.model_ready:
        st.markdown('<span class="status-badge status-ready">✅ Model Ready</span>', unsafe_allow_html=True)
    elif st.session_state.processing:
        st.markdown('<span class="status-badge status-training">⏳ Training in progress...</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">💤 Idle — Paste a URL to start</span>', unsafe_allow_html=True)

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
# Main Content
# ─────────────────────────────────────────────
col_chat, col_info = st.columns([3, 1], gap="large")

# ── RIGHT: Info / Logs ──────────────────────
with col_info:
    if st.session_state.doc_text:
        words = len(st.session_state.doc_text.split())
        n_chunks = len(st.session_state.chunks)
        st.markdown(f"""
<div class="param-card">
  <div class="section-title">📊 Document Stats</div>
  <div style="color:#e2e8f0;font-size:0.85rem">
    📝 <b>{words:,}</b> words extracted<br>
    🧩 <b>{n_chunks}</b> chunks created<br>
    🔗 Type: <b>{(st.session_state.source_type or "—").upper()}</b>
  </div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.log_messages or st.session_state.processing:
        st.markdown('<div class="section-title">🖥️ Training Log</div>', unsafe_allow_html=True)
        log_html = "<br>".join(st.session_state.log_messages[-25:]) or "Waiting..."
        st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)
        if st.session_state.processing:
            time.sleep(2)
            st.rerun()

    if st.session_state.model_ready:
        st.markdown(f"""
<div class="param-card" style="margin-top:12px">
  <div class="section-title">⚙️ Trained Config</div>
  <div style="color:#94a3b8;font-size:0.78rem;line-height:1.8">
    Method: <b style="color:#6c63ff">{method}</b><br>
    Model: <code style="color:#3ecfcf">{model_name.split("/")[-1]}</code><br>
    Rank r: <b>{r}</b> | Alpha: <b>{lora_alpha}</b><br>
    Steps: <b>{max_steps}</b> | LR: <b>{learning_rate:.0e}</b>
  </div>
</div>
""", unsafe_allow_html=True)

# ── LEFT: Chat ─────────────────────────────
with col_chat:
    st.markdown('<div class="section-title">💬 Chat</div>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        if not st.session_state.model_ready:
            lock_msg = (
                "⏳ Training in progress... check the log panel →"
                if st.session_state.processing
                else "🔒 Chat is disabled until fine-tuning completes."
            )
            sub_msg = (
                "The log panel on the right shows live progress."
                if st.session_state.processing
                else "Paste a URL in the sidebar and click <b>Process & Fine-Tune</b>."
            )
            st.markdown(f"""
<div class="chat-disabled-overlay">
    <div style="font-size:2rem;margin-bottom:10px">{'⏳' if st.session_state.processing else '🔒'}</div>
    <div style="font-size:1rem;color:#64748b">{lock_msg}</div>
    <div style="font-size:0.82rem;margin-top:8px;color:#475569">{sub_msg}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="chat-disabled-overlay" style="border-color:#6c63ff55">
    <div style="font-size:2rem;margin-bottom:10px">🚀</div>
    <div style="font-size:1rem;color:#6c63ff">Model is ready!</div>
    <div style="font-size:0.82rem;margin-top:8px;color:#94a3b8">
        Ask anything about the document you loaded.
    </div>
</div>
""", unsafe_allow_html=True)
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
        send_btn = st.button("Send ➤", disabled=not st.session_state.model_ready,
                             use_container_width=True, type="primary")

    if send_btn and user_query.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_query.strip()})
        with st.spinner("Generating response..."):
            response = generate_response(user_query.strip())
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
