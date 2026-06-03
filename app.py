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
# Thread-safe shared state (plain dict)
# st.session_state is NOT accessible from background threads
# ─────────────────────────────────────────────
if "_ts" not in st.session_state:
    st.session_state["_ts"] = {
        "log_messages":    [],
        "model_ready":     False,
        "training_done":   False,
        "processing":      False,
        "deps_installed":  False,
        "doc_text":        "",
        "chunks":          [],
        "source_type":     None,
        "trained_model":   None,
        "trained_tokenizer": None,
    }

_ts = st.session_state["_ts"]   # shortcut — thread-safe plain dict

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    _ts["log_messages"].append(f"[{ts}] {msg}")


def pip_install(name: str, *args):
    """Install a package only if not already importable."""
    try:
        __import__(name.replace("-", "_").split("[")[0])
        log(f"  ✅ {name} already available")
        return True
    except ImportError:
        pass
    log(f"  ⬇️  Installing {name}...")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--user"] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            log(f"  ✅ {name} installed")
            return True
        else:
            log(f"  ⚠️  {name}: {r.stderr.strip()[-150:]}")
            return False
    except subprocess.TimeoutExpired:
        log(f"  ⏱️  {name} timed out")
        return False


def install_ml_deps():
    log("📦 Checking / installing ML dependencies...")
    pip_install("sentencepiece")
    pip_install("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    pip_install("bitsandbytes")
    # Do NOT install unsloth — use transformers+peft directly
    _ts["deps_installed"] = True
    log("✅ ML dependencies ready!")


def split_into_chunks(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    return [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]


def extract_text_from_pdf_url(url: str) -> str:
    from pypdf import PdfReader
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = PdfReader(io.BytesIO(resp.content))
    return "".join(page.extract_text() or "" for page in reader.pages)


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
    """Background thread — uses _ts dict, never st.session_state."""
    try:
        # ── Step 0: deps ──────────────────────────
        if not _ts["deps_installed"]:
            install_ml_deps()

        # ── Step 1: extract text ──────────────────
        log("🔍 Detecting URL type...")
        url_type = detect_url_type(url)
        _ts["source_type"] = url_type
        log(f"📄 Source: {url_type.upper()}")

        log("📥 Extracting text...")
        text = extract_text_from_pdf_url(url) if url_type == "pdf" else extract_text_from_website(url)

        if not text or len(text.strip()) < 100:
            log("❌ Could not extract meaningful text. Check the URL.")
            _ts["processing"] = False
            return
        _ts["doc_text"] = text
        log(f"✅ Extracted {len(text.split()):,} words")

        # ── Step 2: chunk ─────────────────────────
        chunk_size = params.get("chunk_size", 500)
        log(f"✂️  Chunking (size={chunk_size})...")
        chunks = split_into_chunks(text, chunk_size)
        _ts["chunks"] = chunks
        log(f"✅ {len(chunks)} chunks ready")

        # ── Step 3: dataset ───────────────────────
        log("📦 Building dataset...")
        from datasets import Dataset
        alpaca_prompt = (
            "### Instruction:\nUnderstand and remember the following text.\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\nUnderstood and memorized."
        )
        hf_dataset = Dataset.from_list([
            {"text": alpaca_prompt.format(input=chunk)} for chunk in chunks
        ])
        log(f"✅ Dataset ready: {len(hf_dataset)} examples")

        # ── Step 4: load model ────────────────────
        model_name  = params.get("model_name", "facebook/opt-125m")
        max_seq_len = params.get("max_seq_length", 512)
        load_4bit   = (method == "QLORA")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        log(f"🤗 Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if load_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_cfg, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        log("✅ Model loaded")

        # ── Step 5: apply LoRA adapter ────────────
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

        if load_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            r=params.get("r", 16),
            lora_alpha=params.get("lora_alpha", 16),
            lora_dropout=params.get("lora_dropout", 0.0),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        trainable, total = model.get_nb_trainable_parameters()
        log(f"✅ {method} adapter: {trainable:,} trainable / {total:,} total params")

        # ── Step 6: train ─────────────────────────
        log("🏋️  Training started...")
        from trl import SFTTrainer
        from transformers import TrainingArguments

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_len,
            args=TrainingArguments(
                per_device_train_batch_size=params.get("batch_size", 1),
                gradient_accumulation_steps=params.get("gradient_accumulation_steps", 4),
                warmup_steps=params.get("warmup_steps", 5),
                max_steps=params.get("max_steps", 60),
                learning_rate=params.get("learning_rate", 2e-4),
                output_dir="./fine_tuned_model",
                logging_steps=10,
                fp16=torch.cuda.is_available(),
                bf16=False,
                optim="adamw_torch",
                report_to="none",
            ),
        )
        trainer.train()
        log("✅ Training complete!")

        # ── Step 7: inference mode ────────────────
        model.eval()
        _ts["trained_model"]     = model
        _ts["trained_tokenizer"] = tokenizer
        _ts["training_done"]     = True
        _ts["model_ready"]       = True
        log("🚀 Model ready — start chatting!")

    except Exception as e:
        import traceback
        log(f"❌ Error: {str(e)}")
        log(traceback.format_exc()[-400:])
    finally:
        _ts["processing"] = False


def generate_response(query: str) -> str:
    model     = _ts["trained_model"]
    tokenizer = _ts["trained_tokenizer"]
    if not model or not tokenizer:
        return "⚠️ Model not loaded."
    import torch
    prompt = f"### Instruction:\n{query}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip() if "### Response:" in decoded else decoded.strip()


# ─────────────────────────────────────────────
# CSS
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
    padding: 4px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;
}
.status-ready    { background:#1a3a2a; color:#4ade80; border:1px solid #4ade80; }
.status-training { background:#2a2a1a; color:#fbbf24; border:1px solid #fbbf24; }
.status-idle     { background:#1a1d27; color:#94a3b8; border:1px solid #2d2f3e; }
.chat-user {
    background:#6c63ff22; border:1px solid #6c63ff55;
    border-radius:16px 16px 4px 16px; padding:12px 16px;
    margin:8px 0; max-width:82%; margin-left:auto; color:#e2e8f0;
}
.chat-bot {
    background:#1e2435; border:1px solid #2d2f3e;
    border-radius:16px 16px 16px 4px; padding:12px 16px;
    margin:8px 0; max-width:82%; color:#e2e8f0;
}
.chat-label { font-size:0.72rem; color:#64748b; margin-bottom:4px;
    text-transform:uppercase; letter-spacing:0.05em; }
.log-container {
    background:#0a0c12; border:1px solid #1e2435; border-radius:8px;
    padding:12px 14px; font-family:'Courier New',monospace; font-size:0.73rem;
    color:#4ade80; max-height:260px; overflow-y:auto; white-space:pre-wrap; line-height:1.5;
}
.chat-locked {
    background:#0f111788; border:1px dashed #2d2f3e; border-radius:12px;
    padding:32px; text-align:center;
}
.section-title {
    font-size:0.73rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.08em; color:#6c63ff; margin-bottom:10px;
}
.param-card { background:#1a1d27; border:1px solid #2d2f3e;
    border-radius:10px; padding:14px; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span style="font-size:2rem;margin-right:12px">🤖</span>
  <span style="vertical-align:middle">
    <h1 style="display:inline;font-size:1.6rem">Fine-Tune Chat</h1>
    <p>Load any URL → Fine-tune with LoRA / QLoRA → Chat with your data</p>
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">📎 Data Source</div>', unsafe_allow_html=True)
    url_input = st.text_input("URL", placeholder="https://example.com/paper.pdf",
                              label_visibility="collapsed")

    st.markdown('<div class="section-title" style="margin-top:16px">⚡ Method</div>', unsafe_allow_html=True)
    method = st.radio("Method", ["LORA", "QLORA"], horizontal=True, label_visibility="collapsed")

    st.markdown(f'<div class="section-title" style="margin-top:16px">🎛️ {method} Parameters</div>',
                unsafe_allow_html=True)

    with st.expander("Model & Data", expanded=True):
        model_name = st.selectbox("HuggingFace Model", [
            "facebook/opt-125m",           # tiny, fast, great for testing
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "microsoft/phi-1_5",
            "unsloth/Llama-3.2-1B",
            "unsloth/Llama-3.2-3B",
            "unsloth/mistral-7b-bnb-4bit",
        ])
        chunk_size     = st.select_slider("Chunk Size (words)", options=[200,300,400,500,600,800,1000], value=500)
        max_seq_length = st.select_slider("Max Seq Length",     options=[256,512,1024,2048],           value=512)

    with st.expander("LoRA Adapter", expanded=True):
        r            = st.select_slider("Rank (r)",    options=[4,8,16,32,64], value=16)
        lora_alpha   = st.select_slider("LoRA Alpha",  options=[4,8,16,32,64], value=16)
        lora_dropout = st.slider("LoRA Dropout", 0.0, 0.5, 0.0, 0.05)

    with st.expander("Training", expanded=True):
        batch_size                 = st.select_slider("Batch Size",           options=[1,2,4,8],            value=1)
        gradient_accumulation_steps= st.select_slider("Gradient Accum Steps", options=[1,2,4,8,16],         value=4)
        max_steps                  = st.select_slider("Max Steps",            options=[10,20,50,60,100,200], value=60)
        warmup_steps               = st.select_slider("Warmup Steps",         options=[5,10,20,50],          value=5)
        learning_rate              = st.select_slider("Learning Rate",
            options=[1e-5,5e-5,1e-4,2e-4,5e-4,1e-3], value=2e-4,
            format_func=lambda x: f"{x:.0e}")

    st.markdown("---")
    process_btn = st.button("🚀 Process & Fine-Tune", use_container_width=True,
                            type="primary", disabled=_ts["processing"])
    st.markdown("---")
    if _ts["model_ready"]:
        st.markdown('<span class="status-badge status-ready">✅ Model Ready</span>', unsafe_allow_html=True)
    elif _ts["processing"]:
        st.markdown('<span class="status-badge status-training">⏳ Training...</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-idle">💤 Idle</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Process Button
# ─────────────────────────────────────────────
if process_btn:
    if not url_input.strip():
        st.sidebar.error("⚠️ Please enter a URL first.")
    else:
        _ts.update({
            "processing": True, "training_done": False, "model_ready": False,
            "log_messages": [], "doc_text": "", "chunks": [],
        })
        st.session_state["chat_history"] = []

        params = dict(
            model_name=model_name, chunk_size=chunk_size, max_seq_length=max_seq_length,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps, warmup_steps=warmup_steps, learning_rate=learning_rate,
        )
        threading.Thread(
            target=run_training_pipeline,
            args=(url_input.strip(), method, params),
            daemon=True,
        ).start()
        st.rerun()

# ─────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────
col_chat, col_info = st.columns([3, 1], gap="large")

# ── RIGHT: Info / Log ───────────────────────
with col_info:
    if _ts["doc_text"]:
        st.markdown(f"""
<div class="param-card">
  <div class="section-title">📊 Document Stats</div>
  <div style="color:#e2e8f0;font-size:0.85rem">
    📝 <b>{len(_ts["doc_text"].split()):,}</b> words<br>
    🧩 <b>{len(_ts["chunks"])}</b> chunks<br>
    🔗 <b>{(_ts["source_type"] or "—").upper()}</b>
  </div>
</div>""", unsafe_allow_html=True)

    if _ts["log_messages"] or _ts["processing"]:
        st.markdown('<div class="section-title">🖥️ Training Log</div>', unsafe_allow_html=True)
        log_html = "<br>".join(_ts["log_messages"][-35:]) or "Waiting..."
        st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)
        if _ts["processing"]:
            time.sleep(2)
            st.rerun()

    if _ts["model_ready"]:
        st.markdown(f"""
<div class="param-card" style="margin-top:12px">
  <div class="section-title">⚙️ Config</div>
  <div style="color:#94a3b8;font-size:0.78rem;line-height:1.8">
    Method: <b style="color:#6c63ff">{method}</b><br>
    Model: <code style="color:#3ecfcf">{model_name.split("/")[-1]}</code><br>
    r={r} | α={lora_alpha} | steps={max_steps}
  </div>
</div>""", unsafe_allow_html=True)

# ── LEFT: Chat ──────────────────────────────
with col_chat:
    st.markdown('<div class="section-title">💬 Chat</div>', unsafe_allow_html=True)
    model_ready  = _ts["model_ready"]
    processing   = _ts["processing"]
    chat_history = st.session_state["chat_history"]

    if not chat_history:
        if not model_ready:
            icon = "⏳" if processing else "🔒"
            msg  = "Training in progress — check the log →" if processing else "Chat is disabled until fine-tuning completes."
            sub  = "Live progress in the log panel on the right." if processing else "Paste a URL in the sidebar and click <b>Process & Fine-Tune</b>."
            st.markdown(f"""
<div class="chat-locked">
  <div style="font-size:2rem;margin-bottom:10px">{icon}</div>
  <div style="color:#64748b;font-size:1rem">{msg}</div>
  <div style="color:#475569;font-size:0.82rem;margin-top:8px">{sub}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class="chat-locked" style="border-color:#6c63ff55">
  <div style="font-size:2rem;margin-bottom:10px">🚀</div>
  <div style="color:#6c63ff;font-size:1rem">Model ready — ask anything!</div>
</div>""", unsafe_allow_html=True)
    else:
        for msg in chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-label">You</div><div class="chat-user">{msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-label">🤖 Model</div><div class="chat-bot">{msg["content"]}</div>',
                            unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([5, 1])
    with c1:
        user_query = st.text_input(
            "Q", label_visibility="collapsed",
            placeholder="Ask about the document..." if model_ready else "⚠️ Fine-tune first...",
            disabled=not model_ready, key="chat_input",
        )
    with c2:
        send_btn = st.button("Send ➤", disabled=not model_ready,
                             use_container_width=True, type="primary")

    if send_btn and user_query.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_query.strip()})
        with st.spinner("Generating..."):
            response = generate_response(user_query.strip())
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.rerun()

    if chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
