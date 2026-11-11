"""
DSM Regulation Assistant — live dashboard (step 4d; CR-1, FR-11; Design §4.10).

Start it from the project folder:

    streamlit run app/dashboard.py

It opens in your browser at http://localhost:8501. It uses the same code as the
notebooks (src/dsm_assistant), the index built in 04b and the key in .env.
The key is never shown.

Look: dark "grid control room" theme matching the design canvas (IBM Plex, amber accent).
The colours of Streamlit's own widgets come from .streamlit/config.toml.

CR-8 (5.5): themed message boxes, tab-style page switcher, slim demo bar, visitor sidebar in demo
mode (developer details in "About this build"), plain error messages, footer, phone layout.
How answers are produced is unchanged.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

DASHBOARD_VERSION = "5.5"

# Link shown in the footer. Leave empty to show the footer without a link.
REPO_URL = ""

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dsm_assistant import answer, calculator, index, ingest, llm, retrieve, topic  # noqa: E402

EXAMPLES = [
    ("RATES", "What charges apply to a general seller for under injection when grid frequency is between 49.97 Hz and 50.03 Hz?"),
    ("AMENDMENTS", "What is paid for infirm power injected by a standalone energy storage system before its trial run is complete?"),
    ("SECONDARY SOURCE", "What value of X applies to wind generators in 2028-29?"),
    ("PAST DATE", "What was paid for infirm power injected by a standalone energy storage system before its trial run on 01.06.2025?"),
]

TOPIC_EXAMPLES = ["infirm power", "TRAS", "payment of deviation charges", "volume limits for wind and solar sellers"]

STATUS_TEXT = {"final": "in force", "superseded": "superseded"}
TYPE_TEXT = {"draft": "draft", "em": "draft memorandum", "sor": "statement of reasons", "secondary": "secondary source"}
PILL_CLASS = {"in force": "ok", "secondary source": "warn", "draft": "draft", "draft memorandum": "draft",
              "statement of reasons": "muted", "superseded": "muted"}

WAVE = ('<svg width="24" height="24" viewBox="0 0 26 26" fill="none" aria-hidden="true">'
        '<path d="M2 13c2.2-6 4.4-6 6.6 0s4.4 6 6.6 0 4.4-6 6.6 0" stroke="#F2B544" stroke-width="2.3" '
        'stroke-linecap="round"/></svg>')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{--ground:#0B0F14;--panel:#121820;--panel2:#18202A;--line:#243041;--line2:#1E2733;
      --text:#E8EDF3;--soft:#C9D3DF;--muted:#93A1B3;--accent:#F2B544;--good:#5FD39A}
html, body, .stApp, .stMarkdown, button, input, textarea {font-family:'IBM Plex Sans', system-ui, sans-serif !important}
.stApp{background:var(--ground)}
[data-testid="stSidebar"]{background:#0E131A;border-right:1px solid var(--line2)}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.6rem;max-width:1500px}
.mono{font-family:'IBM Plex Mono', monospace !important}
.dsm-head{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:14px;
          padding-bottom:18px;margin-bottom:8px;border-bottom:1px solid var(--line2)}
.dsm-brand{display:flex;align-items:center;gap:14px}
.dsm-logo{width:42px;height:42px;border-radius:11px;background:var(--panel);border:1px solid var(--line);
          display:flex;align-items:center;justify-content:center}
.dsm-title{font-size:21px;font-weight:600;letter-spacing:-.01em;color:var(--text);line-height:1.2}
.dsm-sub{font-size:13px;color:var(--muted)}
.pills{display:flex;flex-wrap:wrap;gap:8px}
.pill{font-family:'IBM Plex Mono', monospace;font-size:12px;padding:5px 10px;border:1px solid var(--line);
      border-radius:999px;color:var(--soft)}
.label{font-family:'IBM Plex Mono', monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.label.accent{color:var(--accent)}
.hero{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:26px 28px;margin:10px 0 18px}
.hero h2{margin:0 0 8px;padding:0;font-size:34px;line-height:1.1;font-weight:700;letter-spacing:-.02em;color:var(--text) !important}
.hero h2 span{color:var(--accent) !important}
.hero p{margin:0;color:var(--soft);font-size:16px;line-height:1.55;max-width:720px}
.ubbl{margin-left:auto;max-width:640px;background:var(--panel2);border:1px solid var(--line);
      border-radius:16px 16px 4px 16px;padding:12px 16px;font-size:15px;line-height:1.5;color:var(--text);width:fit-content}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 22px;margin:10px 0 4px}
.applies{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;margin-bottom:10px}
.applies .txt{font-size:14px;color:var(--soft)}
.body{font-size:16.5px;line-height:1.65;color:var(--text)}
.body p{margin:0 0 8px}
.tag{font-family:'IBM Plex Mono', monospace;font-size:12px;font-weight:600;padding:1px 7px;margin:0 2px;
     border-radius:6px;background:var(--panel2);border:1px solid #2E3A4B;color:var(--accent);white-space:nowrap}
.srcs{margin-top:14px;padding-top:12px;border-top:1px solid var(--line2);display:flex;flex-direction:column;gap:6px}
.src{display:flex;flex-wrap:wrap;align-items:center;gap:10px;font-size:14px;color:var(--text)}
.st{font-family:'IBM Plex Mono', monospace;font-size:11px;padding:2px 8px;border-radius:999px;white-space:nowrap}
.st.ok{background:#10233A;color:#9CCBFF;border:1px solid #23476E}
.st.warn{background:#2A2110;color:#F4CC7A;border:1px solid #5A4416}
.st.draft{background:#2A1426;color:#F2A7DC;border:1px solid #5C2A52}
.st.muted{background:#1A2029;color:var(--muted);border:1px solid var(--line)}
.meta{margin-top:12px;font-family:'IBM Plex Mono', monospace;font-size:12px;color:var(--muted)}
.meta .good{color:var(--good)} .meta .bad{color:#FF8A80}
.refuse{display:flex;gap:14px;align-items:center}
.refuse .ic{width:40px;height:40px;border-radius:50%;border:1px dashed #4A5A6E;flex:none}
.refuse b{font-size:17px;color:var(--text)} .refuse div div{font-size:14px;color:var(--muted)}
.bts-title{display:flex;justify-content:space-between;align-items:center;margin:4px 0 14px}
.bts-title b{font-size:17px;color:var(--text)}
.step{display:flex;gap:12px;margin:6px 0 4px}
.num{width:26px;height:26px;border-radius:50%;border:1px solid var(--accent);color:var(--accent);flex:none;
     font-family:'IBM Plex Mono', monospace;font-size:12px;display:flex;align-items:center;justify-content:center}
.step b{font-size:14px;color:var(--text)}
.kv{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:8px 0 6px 38px}
.kv.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.kv div{background:var(--panel);border:1px solid var(--line2);border-radius:8px;padding:7px 10px}
.kv small{display:block;font-size:11px;color:var(--muted)}
.kv span{font-family:'IBM Plex Mono', monospace;font-size:13px;color:var(--text)}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0 10px 38px;
      font-family:'IBM Plex Mono', monospace;font-size:11px;color:var(--soft)}
.flow span{padding:3px 8px;border-radius:6px;background:var(--panel2)} .flow i{color:#4A5A6E;font-style:normal}
.ok-line{margin:4px 0 0 38px;font-size:13px;color:var(--good)}
.side-rows{display:flex;flex-direction:column;gap:8px;font-size:13px}
.side-rows div{display:flex;justify-content:space-between;gap:8px;color:var(--text)}
.side-rows div span:first-child{color:var(--muted)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
div[data-testid="stButton"] button{text-align:left;justify-content:flex-start;border-radius:10px;min-height:44px}
.brief h2{font-family:'IBM Plex Mono', monospace !important;font-size:12.5px !important;font-weight:600 !important;
          letter-spacing:.12em;text-transform:uppercase;color:var(--accent) !important;
          margin:22px 0 6px !important;padding:0 !important;border-top:1px solid var(--line2);padding-top:14px !important}
.brief h2:first-child{border-top:none;padding-top:0 !important;margin-top:6px !important}
.brief li{margin-bottom:4px}
.calc{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 22px;margin:8px 0}
.calc .big{font-family:'IBM Plex Mono', monospace;font-size:34px;font-weight:600;letter-spacing:-.01em}
.calc .big.rec{color:var(--good)} .calc .big.pay{color:#FFB37A} .calc .big.nil{color:var(--soft)}
.calc .sub{font-size:14px;color:var(--soft);margin-top:4px}
.calc .facts{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.calc .facts span{font-family:'IBM Plex Mono', monospace;font-size:12px;padding:4px 9px;border-radius:7px;
                  background:var(--panel2);color:var(--soft)}
/* CR-8 P-4: Streamlit's message boxes in the theme */
[data-testid="stAlertContainer"]{background:var(--panel) !important;border:1px solid var(--line) !important;
  border-left:3px solid var(--accent) !important;border-radius:10px !important;color:var(--soft) !important}
[data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] div{color:var(--soft) !important}
[data-testid="stAlertContainer"] svg{display:none}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){border-left-color:#FFB37A !important}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){border-left-color:#FF8A80 !important}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){border-left-color:var(--good) !important}
/* CR-8 P-1: the page switcher as tabs */
.st-key-page [role="radiogroup"]{gap:4px !important;border-bottom:1px solid var(--line2);width:100%;flex-wrap:nowrap}
.st-key-page [role="radiogroup"] label{margin:0 !important;padding:10px 16px 9px !important;
  border-bottom:2px solid transparent;cursor:pointer}
.st-key-page [role="radiogroup"] label > div:first-child{display:none !important}
.st-key-page [data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]){display:none !important}
.st-key-page [role="radiogroup"] label p{font-size:14px;font-weight:500;color:var(--muted)}
.st-key-page [role="radiogroup"] label:hover p{color:var(--text)}
.st-key-page [role="radiogroup"] label:has(input:checked){border-bottom-color:var(--accent)}
.st-key-page [role="radiogroup"] label:has(input:checked) p{color:var(--text)}
/* CR-8 P-2: slim demo bar */
.demo-bar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;
  padding:9px 14px;margin:0 0 12px;background:var(--panel);border:1px solid var(--line2);border-radius:10px;
  font-size:13px;color:var(--muted)}
.demo-bar b{color:var(--soft);font-weight:500}
.demo-bar .left{font-family:'IBM Plex Mono', monospace;font-size:12px;padding:3px 10px;border-radius:999px;
  border:1px solid #5A4416;color:var(--accent);white-space:nowrap}
.demo-bar .left.out{border-color:#5C2A2A;color:#FF8A80}
/* CR-8 P-3: visitor sidebar */
.side-h{font-size:14px;font-weight:600;color:var(--text);margin:2px 0 6px}
.side-p{font-size:13px;line-height:1.55;color:var(--soft);margin:0 0 14px}
.side-p ol, .side-p ul{margin:0;padding-left:18px} .side-p li{margin-bottom:3px}
.hint{font-size:12px;color:var(--muted);margin:2px 0 10px}
/* CR-8 P-7: footer */
.dsm-foot{margin-top:36px;padding-top:14px;border-top:1px solid var(--line2);display:flex;flex-wrap:wrap;
  justify-content:space-between;gap:8px;font-size:12px;color:var(--muted)}
.dsm-foot a{color:var(--soft) !important;text-decoration:none;border-bottom:1px solid var(--line)}
.dsm-foot a:hover{color:var(--accent) !important}
/* CR-8 P-8: phone screens */
@media (max-width: 640px){
  .block-container{padding-top:1rem;padding-left:1rem;padding-right:1rem}
  .hero{padding:18px 18px;border-radius:12px}
  .hero h2{font-size:24px}
  .hero p{font-size:14.5px}
  .dsm-title{font-size:18px}
  .pills{display:none}
  .ubbl{max-width:92%}
  .st-key-page [role="radiogroup"] label{padding:8px 10px 7px !important}
  .calc .big{font-size:28px}
}
</style>
"""


# ---------------------------------------------------------------------------
# Loaded once per session (Design §4.10, performance)
# ---------------------------------------------------------------------------
SECRET_KEYS = ("GEMINI_API_KEY", "GROQ_API_KEY", "DEMO_MODE")


def secrets_to_env() -> None:
    """On Streamlit Community Cloud the keys live in the app's Secrets, not in a .env file.
    Copy them into the environment, where the LLM clients look. No secrets file locally = nothing to do."""
    try:
        for k in SECRET_KEYS:
            if not os.environ.get(k) and k in st.secrets:
                os.environ[k] = str(st.secrets[k])
    except Exception:                                    # no secrets.toml: running locally with .env
        pass


def build_clients(settings: dict):
    """Ask client and briefing client.
    Default: llm.make_client(settings) for both (Gemini models in order, then Groq if its key is set).
    With `llm.fast_fallback: groq` and a Groq key (public demo): questions go flash-lite → Groq → the
    slower Gemini models, so a busy free tier costs seconds, not a minute. Briefings (about 15–20k input
    tokens, more than Groq's free per-minute limit) stay on Gemini only."""
    models = list(settings.get("gemini_models") or [])
    if settings.get("fast_fallback") == "groq" and os.environ.get("GROQ_API_KEY") and models:
        common = dict(temperature=settings["temperature"], max_output_tokens=settings["max_output_tokens"],
                      max_retries=settings["max_retries"])
        gem = lambda ms: llm.GeminiClient(ms, thinking_level=settings.get("thinking_level", "low"), **common)
        groq = llm.GroqClient(settings["groq_model"], **common)
        rest = models[1:]
        ask_client = llm.FallbackClient(gem(models[:1]), llm.FallbackClient(groq, gem(rest)) if rest else groq)
        brief_client = llm.FallbackClient(gem(models), None)
        return ask_client, brief_client
    client = llm.make_client(settings)
    return client, client


@st.cache_resource
def usage_today() -> dict:
    """Questions asked today across all visitors (public demo cap). Lives as long as the server process;
    a restart resets it, which errs on the side of the visitor."""
    return {"day": date.today(), "count": 0}


def demo_settings(config: dict) -> dict:
    d = dict(config.get("demo", {}) or {})
    env = os.environ.get("DEMO_MODE", "").strip().lower()
    if env:
        d["enabled"] = env in ("1", "true", "yes", "on")
    d.setdefault("enabled", False)
    d.setdefault("per_visitor", 10)
    d.setdefault("per_day", 150)
    d.setdefault("brief_cost", 2)
    return d


def demo_allow(assistant, cost: int) -> str | None:
    """None = go ahead (and count it); otherwise the message to show instead of calling the model."""
    demo = assistant["demo"]
    if not demo["enabled"]:
        return None
    used = st.session_state.get("demo_used", 0)
    day = usage_today()
    if day["day"] != date.today():
        day["day"], day["count"] = date.today(), 0
    if used + cost > demo["per_visitor"]:
        return (f"This public demo allows {demo['per_visitor']} questions per visitor (a topic brief counts as "
                f"{demo['brief_cost']}). The DSM calculator page has no limit.")
    if day["count"] + cost > demo["per_day"]:
        return "The demo's daily limit has been reached. Please try again tomorrow; the DSM calculator page still works."
    st.session_state["demo_used"] = used + cost
    day["count"] += cost
    return None


def ensure_index(chunks):
    """The laptop has a built index (notebook 04b). The public build ships only the passage vectors
    (data/embeddings.npy, half precision, about 1 MB, because a Chroma file is over GitHub's 25 MB upload
    limit), and the index is built from them here in seconds, with no embedding of passages."""
    idx, npy, ids_file = ROOT / "index", ROOT / "data" / "embeddings.npy", ROOT / "data" / "embeddings_ids.json"
    try:
        col = index.open_vector_index(idx)
        if col.count() == len(chunks):
            return col
    except Exception:                                    # no index folder yet
        pass
    if not npy.exists():
        raise RuntimeError(f"No index matches chunks.jsonl ({len(chunks)} chunks). Rerun notebook 04b.")
    import numpy as np
    vectors = np.load(npy).astype("float32")
    ids = json.loads(ids_file.read_text(encoding="utf-8"))
    if ids != [c.chunk_id for c in chunks] or len(vectors) != len(chunks):
        raise RuntimeError("data/embeddings.npy does not match data/chunks.jsonl. Rerun notebook 07.")

    class Precomputed:
        name = index.DEFAULT_MODEL
        def embed_passages(self, texts):
            return vectors

    return index.build_vector_index(chunks, idx, Precomputed(), log=lambda *a: None)


@st.cache_resource(show_spinner="Loading the index and the embedding model (first time only)…")
def load_assistant():
    secrets_to_env()
    keys = llm.load_env(ROOT)
    settings = llm.load_llm_settings(ROOT / "config.yaml")
    chunks = ingest.read_jsonl(ROOT / "data" / "chunks.jsonl")
    collection = ensure_index(chunks)
    retriever = retrieve.Retriever(chunks, collection, index.load_embedder())
    client, brief_client = build_clients(settings)
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    return {"top_k": int(config.get("answer", {}).get("top_k", 5)), "retriever": retriever, "client": client,
            "brief_client": brief_client, "demo": demo_settings(config),
            "settings": settings, "chunks": len(chunks), "docs": len({c.doc_id for c in chunks}), "keys": keys}


def provider_line(client) -> str:
    line = client.primary.provider
    if getattr(client, "fallback", None):
        line += f" (backup: {client.fallback.provider})"
    return line


# ---------------------------------------------------------------------------
# CR-8: plain error messages, demo bar, footer
# ---------------------------------------------------------------------------
QUOTA_WORDS = ("quota", "429", "rate limit", "rate_limit", "ratelimit", "resource_exhausted",
               "resource exhausted", "too many requests", "exhausted")
NETWORK_WORDS = ("timeout", "timed out", "connection", "network", "unreachable", "503", "502", "504",
                 "unavailable", "temporarily", "dns", "ssl")


def friendly_error(e: Exception) -> str:
    """A plain message for a visitor. The raw error goes to the server log only (CR-8 P-5)."""
    print(f"[dashboard] {type(e).__name__}: {e}", file=sys.stderr)
    text = f"{type(e).__name__} {e}".lower()
    if any(w in text for w in QUOTA_WORDS):
        return "The free model limit is reached for now. Please try again in a few minutes."
    if any(w in text for w in NETWORK_WORDS):
        return "Couldn't reach the model. Please try again."
    return "Something went wrong. Please try again."


def error_text(e: Exception, demo: bool) -> str:
    """Visitors get the plain message; on the laptop (not demo) the detail is added for debugging."""
    msg = friendly_error(e)
    return msg if demo else f"{msg}\n\nDetail (shown only outside demo mode): {e}"


def demo_bar(assistant) -> None:
    demo = assistant["demo"]
    left = max(demo["per_visitor"] - st.session_state.get("demo_used", 0), 0)
    st.markdown(
        '<div class="demo-bar"><span><b>Public demo</b> · answers come only from the indexed CERC documents '
        '· not legal advice</span>'
        f'<span class="left{" out" if left == 0 else ""}">{left} of {demo["per_visitor"]} questions left</span></div>',
        unsafe_allow_html=True)


def footer() -> None:
    link = (f' · <a href="{esc(REPO_URL)}" target="_blank" rel="noopener">Source code on GitHub</a>'
            if REPO_URL else "")
    st.markdown(
        '<div class="dsm-foot"><span>Reference aid, not legal advice. Always check the cited regulation.</span>'
        f'<span>DSM Regulation Assistant{link}</span></div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text), quote=False)


def source_status(chunk) -> str:
    m = chunk.meta
    return TYPE_TEXT.get(m.get("doc_type"), STATUS_TEXT.get(m.get("status"), m.get("status", "")))


def pill(status: str) -> str:
    return f'<span class="st {PILL_CLASS.get(status, "muted")}">{esc(status)}</span>'


TAG_GROUP = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def answer_html(text: str) -> tuple[str, str]:
    """Split off the 'Applies:' line and turn [S1, S3] into styled tags. Returns (applies, body)."""
    text = text.strip()
    applies = ""
    if text.lower().startswith("applies:"):
        first, _, text = text.partition("\n")
        applies = first.split(":", 1)[1].strip()
    body = esc(text.strip())
    body = TAG_GROUP.sub(lambda m: " ".join(f'<span class="tag">{t.strip()}</span>' for t in m.group(1).split(",")), body)
    return applies, body


def passages_table(a: answer.Answer) -> pd.DataFrame:
    rows = []
    for i, h in enumerate(a.hits, 1):
        tag = f"S{i}"
        rows.append({
            "tag": tag,
            "used in answer": "✔" if tag in a.cited else "",
            "fused score": round(h.score, 4),
            "meaning rank": str(h.vector_rank) if h.vector_rank is not None else "–",
            "keyword rank": str(h.keyword_rank) if h.keyword_rank is not None else "–",
            "how it got here": "amendment expansion" if h.expanded else "ranking",
            "source": answer.citation(h.chunk),
            "status": source_status(h.chunk),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DSM calculator display (used by the chat and the calculator page)
# ---------------------------------------------------------------------------
def calc_card(res: dict, title: str = "DSM calculator") -> None:
    """res: calculator.Result.as_dict()."""
    total, outcome = res["total_rs"], res["outcome"]
    cls = {"receivable": "rec", "payable": "pay"}.get(outcome, "nil")
    direction = res["direction"].replace("_", " ")
    pct = f' ({res["deviation_pct"]:+.2f}%)' if res.get("deviation_pct") is not None else ""
    facts = [res["entity"], res["date"], f'{res["deviation_mwh"]:+.3f} MWh{pct}', direction]
    if res.get("frequency_hz") is not None:
        facts.append(f'{res["frequency_hz"]:.2f} Hz')
    st.markdown(
        f'<div class="calc"><span class="label accent">{esc(title)}</span>'
        f'<div class="big {cls}">₹{abs(total):,.2f}</div>'
        f'<div class="sub">{esc(outcome)} by the {"buyer" if "Buyer" in res["entity"] else "entity"} for this '
        f'15-minute block</div><div class="facts">{"".join(f"<span>{esc(x)}</span>" for x in facts)}</div></div>',
        unsafe_allow_html=True)
    if res["slabs"]:
        st.dataframe(pd.DataFrame([{"part": x["part"], "MWh": x["mwh"], "rate": x["rate"],
                                    "₹/kWh": x["rate_rs_per_kwh"], "amount ₹": x["amount_rs"],
                                    "clause": x["clause"]} for x in res["slabs"]]), hide_index=True)
    for n in res.get("notes", []):
        st.caption("Note: " + n)


# ---------------------------------------------------------------------------
# The answer (left column)
# ---------------------------------------------------------------------------
def show_answer(a: answer.Answer) -> None:
    applies, body = answer_html(a.text)
    parts = ['<div class="card">']
    if a.refused:
        parts.append('<div class="refuse"><span class="ic"></span><div><b>Not found in the documents.</b>'
                     '<div>The assistant refuses instead of guessing when the passages do not answer the question.</div>'
                     '</div></div>')
    else:
        if applies:
            parts.append(f'<div class="applies"><span class="label accent">Applies</span>'
                         f'<span class="txt">{esc(applies)}</span></div>')
        parts.append(f'\n\n<div class="body">\n\n{body}\n\n</div>\n\n')
        if a.cited:
            rows = "".join(
                f'<div class="src">{" ".join(f"<span class=tag>{t}</span>" for t in tags)}'
                f'<span>{esc(cite)}</span>{pill(source_status(a.sources[tags[0]]))}</div>'
                for tags, cite in a.source_list())
            parts.append(f'<div class="srcs"><span class="label">Sources</span>{rows}</div>')

    check = ('<span class="good">refused cleanly</span>' if a.refused else
             '<span class="good">citations valid</span>' if a.ok else '<span class="bad">check failed</span>')
    notes = f" · {esc('; '.join(a.notes))}" if a.notes else ""
    parts.append(f'<div class="meta">{esc(a.model or a.provider)} · {a.seconds:.1f} s · {check}{notes}</div></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)
    for c in a.tool_calls:
        res = c.get("result", {})
        if res.get("ok"):
            calc_card(res["result"], "Computed by the DSM calculator")
        else:
            st.warning(f"DSM calculator: {res.get('message', res.get('error', 'error'))}")

    problems = []
    if a.invalid_citations:
        problems.append(f"cites passages it was not given: {', '.join(a.invalid_citations)}")
    if a.uncited:
        problems.append("the answer has no citations")
    if problems:
        st.warning("Check failed: " + "; ".join(problems))
    if a.uses_draft:
        st.info("This answer uses a **draft**: a proposal, not law.")
    if a.uses_secondary:
        st.info("This answer uses a **secondary source**: a third-party summary, not the official text.")


# ---------------------------------------------------------------------------
# Behind the scenes (right column)
# ---------------------------------------------------------------------------
def show_behind_the_scenes(a: answer.Answer) -> None:
    an = a.analysis
    as_of = an.as_of.strftime('%d.%m.%Y') if an.as_of else "all versions"
    st.markdown(
        f'<div class="bts-title"><b>Behind the scenes</b><span class="mono" style="color:#93A1B3;font-size:12px">'
        f'{a.seconds:.1f} s</span></div>'
        f'<div class="step"><span class="num">1</span><b>How the question was understood</b></div>'
        f'<div class="kv">'
        f'<div><small>Rules as of</small><span>{as_of}</span></div>'
        f'<div><small>Date in question</small><span>{an.date.strftime("%d.%m.%Y") if an.date else "none"}</span></div>'
        f'<div><small>Drafts</small><span>{"allowed" if an.wants_draft else "excluded"}</span></div>'
        f'<div><small>Comparison</small><span>{"yes" if an.compare else "no"}</span></div>'
        f'</div>'
        f'<div class="step"><span class="num">2</span><b>Passages found and given to the model</b></div>'
        f'<div class="flow"><span>version filter</span><i>→</i><span>meaning (bge)</span><i>+</i>'
        f'<span>keywords (BM25)</span><i>→</i><span>fused (RRF)</span><i>→</i><span>top {len(a.hits)}</span></div>',
        unsafe_allow_html=True)

    if a.hits:
        st.dataframe(passages_table(a), hide_index=True,
                     column_config={"source": st.column_config.TextColumn(width="large")})
        tabs = st.tabs([f"S{i}" + (" ✔" if f"S{i}" in a.cited else "") for i in range(1, len(a.hits) + 1)])
        for tab, h in zip(tabs, a.hits):
            with tab:
                st.markdown(f"**{h.chunk.breadcrumb}**")
                st.caption(answer.citation(h.chunk))
                st.text(h.chunk.text)
    else:
        st.caption("No passage passed the version filter, so the model was not asked.")

    if a.tool_calls:
        st.markdown('<div class="step"><span class="num">⚙</span><b>Tool called by the model: dsm_calculator</b></div>',
                    unsafe_allow_html=True)
        for c in a.tool_calls:
            st.caption("Inputs the model passed (taken from the question):")
            st.json(c.get("args", {}), expanded=False)
            res = c.get("result", {})
            st.caption(("Result: ₹{:,.2f} {}".format(abs(res["result"]["total_rs"]), res["result"]["outcome"])
                        if res.get("ok") else f"Result: {res.get('message', res.get('error'))}"))

    tokens = a.thinking_tokens if a.thinking_tokens else "–"
    retries = "; ".join(a.notes) if a.notes else "none"
    check = ("Refused cleanly instead of guessing" if a.refused else
             "Citations checked: built by code from the index, so the model can't invent a clause" if a.ok else
             "Citation check failed")
    st.markdown(
        f'<div class="step"><span class="num">3</span><b>The model</b></div>'
        f'<div class="kv three">'
        f'<div><small>Model</small><span>{esc(a.model or a.provider)}</span></div>'
        f'<div><small>Model time</small><span>{a.llm_seconds:.1f} s</span></div>'
        f'<div><small>Thinking</small><span>{tokens}</span></div>'
        f'</div>'
        f'<div class="kv"><div style="grid-column:1/-1"><small>Retries or fallback</small><span>{esc(retries)}</span></div></div>'
        f'<div class="ok-line">✔ {check}</div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Calculator page (CR-4)
# ---------------------------------------------------------------------------
def calculator_page() -> None:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown('<span class="label">Time block (15 minutes)</span>', unsafe_allow_html=True)
        entity = st.selectbox("Entity", list(calculator.ENTITIES), format_func=calculator.ENTITIES.get,
                              key="calc-entity")
        block_date = st.date_input("Date", value=date.today(), format="DD.MM.YYYY", key="calc-date",
                                   min_value=date(2024, 9, 16))
        args: dict = {"entity": entity, "date": block_date}
        mode = None
        if entity in ("ess", "psp"):
            mode = st.radio("Mode", ["discharging", "charging"], horizontal=True, key="calc-mode")
            args["ess_mode"] = mode
        if entity == "infirm":
            args["infirm_stage"] = st.radio("Stage", ["before_trial", "after_trial"], horizontal=True, key="calc-stage",
                                            format_func={"before_trial": "Before trial run complete",
                                                         "after_trial": "Scheduled, after trial run"}.get)
            if args["infirm_stage"] == "before_trial":
                args["infirm_unit"] = st.selectbox("Unit type", ["thermal", "ess", "other"], key="calc-unit")
            else:
                args["infirm_seller"] = st.selectbox("Charged as", ["general", "ws"], key="calc-iseller",
                                                     format_func={"general": "General seller", "ws": "WS seller"}.get)
        drawal = entity == "buyer" or mode == "charging"
        word = "drawal" if drawal else "injection"
        c1, c2 = st.columns(2)
        if not (entity == "infirm" and args.get("infirm_stage") == "before_trial"):
            args["scheduled_mw"] = c1.number_input(f"Scheduled {word} (MW)", min_value=0.0, value=100.0,
                                                   step=1.0, key="calc-sched")
        else:
            args["scheduled_mw"] = 0.0
        args["actual_mw"] = c2.number_input(f"Actual {word} (MW)", min_value=0.0, value=90.0, step=1.0,
                                            key="calc-actual")

        if entity == "ws":
            args["ws_type"] = st.selectbox("Source", ["solar", "wind", "hybrid"], key="calc-wstype")
        if entity == "infirm" and args.get("infirm_seller") == "ws":
            args["ws_type"] = st.selectbox("Source", ["solar", "wind", "hybrid"], key="calc-iwstype")
        needs_avc = entity == "ws" or (entity == "psp" and mode == "charging" and block_date < date(2026, 4, 1)) \
            or (entity == "infirm" and args.get("infirm_seller") == "ws")
        if needs_avc:
            c1, c2 = st.columns(2)
            args["available_capacity_mw"] = c1.number_input("Available capacity (MW)", min_value=0.0, value=100.0,
                                                            step=1.0, key="calc-avc")
            if entity == "ws":
                args["connectivity_mw"] = c2.number_input("Connectivity granted (MW, optional)", min_value=0.0,
                                                          value=None, step=1.0, key="calc-conn")

        freq_needed = entity in ("general", "ess", "buyer") or (entity == "infirm") or \
            (entity == "psp" and not (mode == "charging" and block_date < date(2026, 4, 1)))
        if entity == "ws":
            with st.expander("Project treated at par with general sellers? (Reg. 8(4A), from 31.08.2026)"):
                route = st.radio("Route", ["none", "bid", "other"], horizontal=True, key="calc-route",
                                 format_func={"none": "not applicable", "bid": "bidding route", "other": "other"}.get)
                if route == "bid":
                    args["ws_route"], args["bid_date"] = "bid", st.date_input("Tender / bid date", key="calc-bid",
                                                                              format="DD.MM.YYYY")
                elif route == "other":
                    args["ws_route"], args["cod_date"] = "other", st.date_input("Commercial operation date",
                                                                                key="calc-cod", format="DD.MM.YYYY")
                if route != "none":
                    freq_needed = True
            if block_date >= date(2026, 4, 1) and st.checkbox("Enter X myself", key="calc-xown"):
                args["x_percent"] = st.number_input("X (%)", min_value=0.0, max_value=100.0, value=100.0,
                                                    key="calc-x")
        if freq_needed:
            args["frequency_hz"] = st.number_input("Average frequency (Hz)", min_value=49.0, max_value=51.0,
                                                   value=50.00, step=0.01, format="%.2f", key="calc-f")

        st.markdown('<span class="label">Rates (₹/kWh)</span>', unsafe_allow_html=True)
        if entity in ("general", "ror", "ess") or (entity == "psp" and not (mode == "charging" and
                                                                             block_date < date(2026, 4, 1))):
            if entity == "psp" and block_date >= date(2026, 8, 31) and st.checkbox(
                    "Tariff under Section 62 (energy charge rate applies)", key="calc-s62"):
                args["psp_section_62"] = True
                args["energy_charge_rate"] = st.number_input("Energy charge rate", min_value=0.0, value=3.0,
                                                             step=0.01, key="calc-ecr")
            else:
                args["reference_rate"] = st.number_input("Reference charge rate (RR)", min_value=0.0, value=3.0,
                                                         step=0.01, key="calc-rr")
        if entity in ("ws", "msw") or (entity == "psp" and mode == "charging" and block_date < date(2026, 4, 1)):
            args["contract_rate"] = st.number_input("Contract rate", min_value=0.0, value=3.0, step=0.01,
                                                    key="calc-cr")
        if entity == "infirm" and args.get("infirm_stage") == "after_trial":
            if args.get("infirm_seller") == "ws":
                args["contract_rate"] = st.number_input("Contract rate", min_value=0.0, value=3.0, step=0.01,
                                                        key="calc-icr")
            else:
                args["reference_rate"] = st.number_input("Reference charge rate (RR)", min_value=0.0, value=3.0,
                                                         step=0.01, key="calc-irr")
        if entity == "buyer" or (entity == "infirm" and args.get("infirm_stage") == "before_trial"):
            if st.checkbox("Compute NR from market prices (Reg. 7)", key="calc-nrparts"):
                c1, c2, c3 = st.columns(3)
                dam = c1.number_input("DAM ACP (paise)", min_value=0.0, value=450.0, key="calc-dam")
                rtm = c2.number_input("RTM ACP (paise)", min_value=0.0, value=500.0, key="calc-rtm")
                asc = c3.number_input("AS charge (paise)", min_value=0.0, value=600.0, key="calc-as")
                args["normal_rate"] = calculator.normal_rate_from_components(dam, rtm, asc)
                st.caption(f"NR = ₹{args['normal_rate']:.4f}/kWh (highest of DAM, RTM and their average with AS)")
            else:
                args["normal_rate"] = st.number_input("Normal Rate of charges for deviation (NR)", min_value=0.0,
                                                      value=4.0, step=0.01, key="calc-nr")
        if entity == "buyer":
            args["buyer_category"] = st.selectbox("Buyer category", list(calculator.BUYER_CATEGORIES),
                                                  format_func=calculator.BUYER_CATEGORIES.get, key="calc-bcat")

    with right:
        try:
            result = calculator.calculate(**{k: v for k, v in args.items() if v is not None})
            calc_card(result.as_dict(), "Charge for deviation")
            with st.expander("How it was worked out"):
                st.text(result.summary())
                st.caption("Rules: " + "; ".join(result.clauses) +
                           ". Source: CERC DSM Regulations, 2024 with errata and three amendments "
                           "(docs/05-calculator-rules.md).")
        except calculator.OutOfScope as e:
            st.info(f"Not covered by the calculator: {e}")
        except calculator.CalculatorError as e:
            st.warning(str(e))
        st.caption("One 15-minute block. Amounts: energy (MWh) × rate. Values in ₹/kWh; MW are block averages.")


# ---------------------------------------------------------------------------
# Topic brief page (CR-5, FR-12)
# ---------------------------------------------------------------------------
def brief_body_html(text: str) -> str:
    body = esc(text.strip())
    return TAG_GROUP.sub(lambda m: " ".join(f'<span class="tag">{t.strip()}</span>' for t in m.group(1).split(",")), body)


def show_briefing(b: topic.Briefing) -> None:
    parts = [f'<div class="card"><div class="applies"><span class="label accent">Topic</span>'
             f'<span class="txt">{esc(b.topic)}</span></div>']
    if b.refused:
        parts.append('<div class="refuse"><span class="ic"></span><div><b>Not found in the documents.</b>'
                     '<div>No passage in the documents is about this topic.</div></div></div>')
    else:
        parts.append(f'\n\n<div class="body brief">\n\n{brief_body_html(b.text)}\n\n</div>\n\n')
        if b.cited:
            rows = "".join(
                f'<div class="src">{" ".join(f"<span class=tag>{t}</span>" for t in tags)}'
                f'<span>{esc(cite)}</span>{pill(source_status(b.sources[tags[0]]))}</div>'
                for tags, cite in b.source_list())
            parts.append(f'<div class="srcs"><span class="label">Sources</span>{rows}</div>')
    check = ('<span class="good">refused cleanly</span>' if b.refused else
             '<span class="bad">incomplete</span>' if b.missing_sections else
             '<span class="good">citations valid</span>' if b.ok else '<span class="bad">check failed</span>')
    notes = f" · {esc('; '.join(b.notes))}" if b.notes else ""
    parts.append(f'<div class="meta">{esc(b.model or b.provider)} · {b.seconds:.1f} s · {len(b.hits)} passages · '
                 f'{len(set(b.cited))} cited · {check}{notes}</div></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)
    if b.invalid_citations:
        st.warning(f"Check failed: cites passages it was not given: {', '.join(b.invalid_citations)}")
    if b.missing_sections:
        st.warning("**Incomplete briefing:** the model stopped early. Missing: " + ", ".join(b.missing_sections)
                   + ". Try again, or use a narrower topic.")
    if b.rule_section_misuse:
        st.warning("*Rules in force today* cites a Statement of Reasons, memorandum or draft ("
                   + ", ".join(b.rule_section_misuse) + "): check those rules against the regulation itself.")
    if b.uses_draft:
        st.info("This briefing uses **drafts**: proposals, not law. They appear only under *Proposals (drafts)*.")
    if b.uses_secondary:
        st.info("This briefing uses a **secondary source**: a third-party summary, not the official text.")


def show_topic_passages(b: topic.Briefing) -> None:
    st.markdown('<div class="bts-title"><b>Passages used</b></div>', unsafe_allow_html=True)
    st.caption("Wide search (topic + 4 angles, all versions, drafts and reasons included), grouped by regulation "
               "family, oldest first. ✔ = cited in the briefing.")
    tag_of = {id(chunk): tag for tag, chunk in b.sources.items()}
    for fam, group in b.result.groups:
        st.markdown(f'<span class="label accent">{esc(fam)}</span>', unsafe_allow_html=True)
        rows = []
        for h in group:
            tag = tag_of.get(id(h.chunk), "")
            m = h.chunk.meta
            span = (m["effective_from"] or "–") + (" → " + m["effective_to"] if m["effective_to"] else "")
            rows.append({"tag": tag, "cited": "✔" if tag in b.cited else "", "source": answer.citation(h.chunk),
                         "status": source_status(h.chunk), "applies": span,
                         "how": "amendment expansion" if h.expanded else "search"})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"source": st.column_config.TextColumn(width="large")})


def topic_page(assistant) -> None:
    left, right = st.columns([3, 2], gap="large")
    briefs = st.session_state.setdefault("briefs", [])
    with left:
        if not briefs:
            st.markdown('<div class="hero"><h2>Give a topic. <span>Get everything on it.</span></h2>'
                        '<p>A briefing from the documents: the rules in force today, how they changed and when, '
                        'related rules in the Grid Code, GNA, Ancillary Services and Power Market regulations, the '
                        'Commission\'s reasons, and what drafts proposed. Every fact is cited.</p></div>'
                        '<span class="label">Try a topic</span>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, t in enumerate(TOPIC_EXAMPLES):
                if cols[i % 2].button(t, key=f"topic-example-{i}", use_container_width=True):
                    st.session_state.pending_topic = t
            if assistant["demo"]["enabled"]:
                st.markdown(f'<div class="hint">Each topic brief uses {assistant["demo"]["brief_cost"]} of your '
                            'questions.</div>', unsafe_allow_html=True)
        with st.form("topic-form", clear_on_submit=True):
            t = st.text_input("Topic", placeholder="e.g. infirm power, TRAS, connectivity bank guarantee", key="topic-text")
            submitted = st.form_submit_button("Brief me")
        chosen = (t if submitted else None) or st.session_state.pop("pending_topic", None)
        blocked = demo_allow(assistant, assistant["demo"]["brief_cost"]) if chosen and chosen.strip() else None
        if blocked:
            st.warning(blocked)
        elif chosen and chosen.strip():
            try:
                with st.spinner("Gathering every relevant passage and writing the briefing…"):
                    briefs.insert(0, topic.brief(chosen.strip(), assistant["retriever"], assistant["brief_client"],
                                                 today=date.today()))
                st.rerun()                                      # redraw without the intro
            except llm.LLMError as e:
                st.error(error_text(e, assistant["demo"]["enabled"]))
        if briefs:
            show_briefing(briefs[0])
            if len(briefs) > 1:
                st.caption("Earlier briefings: " + " · ".join(esc(x.topic) for x in briefs[1:6]))
    with right:
        if briefs:
            show_topic_passages(briefs[0])
        else:
            st.markdown('<div class="bts-title"><b>Passages used</b></div>', unsafe_allow_html=True)
            st.caption("Give a topic to see every passage that was gathered, grouped by regulation family.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def header(assistant) -> None:
    st.markdown(
        f'<div class="dsm-head"><div class="dsm-brand"><div class="dsm-logo">{WAVE}</div><div>'
        f'<div class="dsm-title">DSM Regulation Assistant</div>'
        f'<div class="dsm-sub">Grounded answers on CERC\'s Deviation Settlement Mechanism, with a citation for every fact</div>'
        f'</div></div><div class="pills">'
        f'<span class="pill">{assistant["docs"]} documents</span>'
        f'<span class="pill">{assistant["chunks"]} passages</span>'
        f'<span class="pill">hybrid search · RRF</span></div></div>',
        unsafe_allow_html=True)


def status_rows(assistant) -> None:
    """Index, model and key status (no key value is ever shown, only set / not set)."""
    gem = bool(os.environ.get("GEMINI_API_KEY"))
    groq = bool(os.environ.get("GROQ_API_KEY"))
    models = assistant["settings"]["gemini_models"]
    st.markdown(
        '<div class="side-rows">'
        f'<div><span>Index</span><span><b>{assistant["chunks"]}</b> passages · {assistant["docs"]} docs</span></div>'
        f'<div><span>Model</span><span class="mono" style="font-size:12px">{esc(provider_line(assistant["client"]))}</span></div>'
        f'<div><span>First choice</span><span class="mono" style="font-size:12px">{esc(models[0]) if models else "–"}</span></div>'
        f'<div><span>Gemini key</span><span><span class="dot" style="background:{"#5FD39A" if gem else "#FF8A80"}"></span>'
        f'{"set" if gem else "not set"}</span></div>'
        f'<div><span>Groq key</span><span><span class="dot" style="background:{"#5FD39A" if groq else "#4A5A6E"}"></span>'
        f'{"set" if groq else "not set"}</span></div>'
        '</div>',
        unsafe_allow_html=True)
    st.caption(f"Dashboard {DASHBOARD_VERSION} · answer {answer.VERSION} · topic {topic.VERSION} · llm {llm.VERSION}")


def date_controls():
    pick_date = st.checkbox("Answer for a different date",
                            help="Questions without a date are answered for the rules in force on this date. "
                                 "A date written in the question still wins.")
    return st.date_input("Rules in force as of", value=date.today(), format="DD.MM.YYYY") if pick_date else None


def clear_chat_button() -> None:
    if st.button("Clear chat", use_container_width=False):
        st.session_state.history = []
        st.session_state.pop("selected", None)


def sidebar(assistant):
    with st.sidebar:
        if assistant["demo"]["enabled"]:                       # CR-8 P-3: what a visitor needs
            st.markdown(
                '<div class="side-h">How it works</div><div class="side-p"><ol>'
                '<li>Your question is matched to the regulation version in force on its date.</li>'
                '<li>The most relevant clauses are found by meaning and by keywords.</li>'
                '<li>The model answers only from those clauses, and every fact is cited.</li></ol></div>'
                '<div class="side-h">Covers</div><div class="side-p"><ul>'
                '<li>CERC DSM Regulations, past and current, with amendments</li>'
                '<li>Related CERC rules: Grid Code, GNA, Ancillary Services, Power Market</li></ul></div>'
                '<div class="side-h">Doesn\'t cover</div><div class="side-p"><ul>'
                '<li>State (SERC) regulations</li><li>Orders and documents not indexed here</li>'
                '<li>Legal advice</li></ul></div>',
                unsafe_allow_html=True)
            st.markdown('<span class="label">Settings</span>', unsafe_allow_html=True)
            as_of = date_controls()
            clear_chat_button()
            st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            with st.expander("About this build"):
                top_k = st.slider("Passages given to the model", 3, 8, assistant["top_k"])
                status_rows(assistant)
            return as_of, top_k

        st.markdown('<span class="label">Settings</span>', unsafe_allow_html=True)
        as_of = date_controls()
        top_k = st.slider("Passages given to the model", 3, 8, assistant["top_k"])
        clear_chat_button()
        st.markdown('<div style="height:18px"></div><span class="label">Status</span><div style="height:10px"></div>',
                    unsafe_allow_html=True)
        status_rows(assistant)
    return as_of, top_k


def main() -> None:
    st.set_page_config(page_title="DSM Regulation Assistant", page_icon="⚡", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        assistant = load_assistant()
    except Exception as e:                                     # missing index, chunks or package
        st.title("DSM Regulation Assistant")
        demo_env = os.environ.get("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")
        print(f"[dashboard] start-up failed: {type(e).__name__}: {e}", file=sys.stderr)
        st.error("The assistant couldn't start. Please try again in a few minutes." if demo_env
                 else f"The assistant could not start: {e}")
        st.stop()

    header(assistant)
    if assistant["demo"]["enabled"]:
        demo_bar(assistant)
    as_of, top_k = sidebar(assistant)
    history = st.session_state.setdefault("history", [])

    page = st.radio("Page", ["Ask", "Topic brief", "DSM calculator"], horizontal=True,
                    label_visibility="collapsed", key="page")
    if page == "DSM calculator":
        calculator_page()
        footer()
        return
    if page == "Topic brief":
        topic_page(assistant)
        footer()
        return

    chat, panel = st.columns([3, 2], gap="large")

    with chat:
        if not history:
            st.markdown('<div class="hero"><h2>Ask the regulation. <span>Get the clause.</span></h2>'
                        '<p>Answers come only from the CERC documents, for the version in force on the date you ask '
                        'about, with the clause and page for every fact. Drafts and summaries are flagged, and it says '
                        '"not found" instead of guessing.</p></div><span class="label">Try a question</span>',
                        unsafe_allow_html=True)
            cols = st.columns(2)
            for i, (kind, q) in enumerate(EXAMPLES):
                if cols[i % 2].button(f"{kind}  ·  {q}", key=f"example-{i}", use_container_width=True):
                    st.session_state.pending = q
            if assistant["demo"]["enabled"]:
                st.markdown('<div class="hint">Each example uses 1 of your questions.</div>', unsafe_allow_html=True)

        for item in history:
            st.markdown(f'<div class="ubbl">{esc(item["question"])}</div>', unsafe_allow_html=True)
            if item.get("error"):
                st.error(item["error"])
            else:
                show_answer(item["answer"])

    question = st.chat_input("Ask about the DSM regulations…") or st.session_state.pop("pending", None)
    blocked = demo_allow(assistant, 1) if question else None
    if blocked:
        with chat:
            st.warning(blocked)
    elif question:
        with chat:
            st.markdown(f'<div class="ubbl">{esc(question)}</div>', unsafe_allow_html=True)
            try:
                with st.spinner("Searching the documents and asking the model…"):
                    a = answer.ask(question, assistant["retriever"], assistant["client"],
                                   k=top_k, today=date.today(), as_of=as_of)
                history.append({"question": question, "answer": a})
                st.session_state.selected = len(history) - 1
            except llm.LLMError as e:
                history.append({"question": question, "error": error_text(e, assistant["demo"]["enabled"])})
        st.rerun()

    with panel:
        answered = [i for i, item in enumerate(history) if "answer" in item]
        if not answered:
            st.markdown('<div class="bts-title"><b>Behind the scenes</b></div>', unsafe_allow_html=True)
            st.caption("Ask a question to see how it was understood, which passages were found and ranked, "
                       "and how the model answered.")
        else:
            chosen = st.session_state.get("selected", answered[-1])
            if chosen not in answered:
                chosen = answered[-1]
            if len(answered) > 1:
                chosen = st.selectbox("Show details for", answered, index=answered.index(chosen),
                                      format_func=lambda i: f"Q{i + 1}: {history[i]['question'][:70]}")
            show_behind_the_scenes(history[chosen]["answer"])

    footer()


main()
