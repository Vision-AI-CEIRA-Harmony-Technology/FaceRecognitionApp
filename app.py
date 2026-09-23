"""TopoFR demo - 1:1 verification and 1:N identification, CPU vs GPU.

    python -m streamlit run app.py

First run asks for two things: the `.pt` checkpoint and the dataset folder. The
gallery is then enrolled once from that dataset and cached, and the two test
screens open.

Both screens time the pipeline step by step on the device picked in the sidebar.
The checkpoint is loaded directly by PyTorch and CPU and GPU run the identical
model through the identical runtime, so the device is the only variable between
the two measurements.

The 1:N gallery is embedded and normalised at enrollment, so the latency reported
for a search is only what the incoming probe costs: preprocess + embed + compare.
That keeps the 1:1 and 1:N figures on the same basis.
"""
import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import dataset as D
import engine as E
import pipeline as P
from enroll import build as build_gallery

st.set_page_config(
    page_title="TopoFR - CPU vs GPU",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODERN CLEAN DESIGN SYSTEM
# White background, purple and orange modern accents. Clean geometry, minimal.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Tokens ── */
:root {
    --bg:        #ffffff;
    --surface:   #f9fafb;
    --card:      #ffffff;
    --card-alt:  #f3f4f6;
    --purple:    #7c3aed;
    --purple-hi: #8b5cf6;
    --purple-lo: #5b21b6;
    --purple-dim: rgba(124, 58, 237, 0.08);
    --purple-line: rgba(124, 58, 237, 0.25);
    --orange:    #f97316;
    --orange-hi: #fb923c;
    --txt:       #111827;
    --txt-2:     #4b5563;
    --txt-3:     #9ca3af;
    --ok:        #10b981;
    --no:        #ef4444;
    --line:      #e5e7eb;
    --r:         6px;
    --r-lg:      10px;
    --font-head: 'Inter', -apple-system, sans-serif;
    --font-body: 'Inter', -apple-system, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}

/* ── Reset ── */
html, body, [class*="css"] { font-family: var(--font-body) !important; }
.stApp { background: var(--bg) !important; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1160px !important;
}

/* ── Top accent bar ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--purple) 0%, var(--orange) 100%);
    z-index: 9999;
}

/* ── Typography ── */
h1 { font-family: var(--font-head) !important; font-weight: 700 !important;
     font-size: 2rem !important; letter-spacing: -0.02em !important;
     color: var(--txt) !important; }
h2 { font-family: var(--font-head) !important; font-weight: 600 !important;
     font-size: 1.25rem !important; letter-spacing: -0.01em !important;
     color: var(--txt) !important; }
h3 { font-family: var(--font-head) !important; font-weight: 600 !important;
     font-size: 1rem !important; letter-spacing: 0 !important;
     color: var(--purple) !important; }
p, span, div, label { color: var(--txt-2); }
strong, b { color: var(--txt); }
code {
    font-family: var(--font-mono) !important;
    background: var(--purple-dim) !important;
    color: var(--purple-lo) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: .82em !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 2.5rem !important; }
section[data-testid="stSidebar"] h3 {
    font-size: .75rem !important;
    color: var(--txt-3) !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
}

/* ── Radio ── */
div[data-testid="stRadio"] label {
    color: var(--txt-2) !important;
    font-size: .9rem !important;
    padding: 6px 12px !important;
    border-radius: var(--r) !important;
    transition: all .15s;
}
div[data-testid="stRadio"] label:hover {
    background: var(--purple-dim) !important;
    color: var(--purple) !important;
}
div[data-testid="stRadio"] label[data-checked="true"] {
    color: var(--purple) !important;
    background: var(--purple-dim) !important;
    font-weight: 500 !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div { background: var(--purple) !important; }
.stSlider > div > div > div { background: var(--line) !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: var(--font-head) !important;
    font-weight: 500 !important;
    font-size: .85rem !important;
    border-radius: var(--r) !important;
    border: 1px solid var(--line) !important;
    transition: all .15s;
}
.stButton > button[kind="primary"] {
    background: var(--purple) !important;
    border-color: var(--purple) !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--purple-lo) !important;
    border-color: var(--purple-lo) !important;
}
.stButton > button:hover { border-color: var(--purple) !important; color: var(--purple) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    color: var(--txt-3) !important;
    padding: 10px 24px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px;
    transition: color .2s, border-color .2s;
    background: none !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--txt) !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--purple) !important;
    border-bottom-color: var(--purple) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ── File uploaders ── */
section[data-testid="stFileUploader"] > div {
    border: 1px dashed var(--line) !important;
    border-radius: var(--r) !important;
    background: var(--surface) !important;
    transition: all .2s;
}
section[data-testid="stFileUploader"] > div:hover {
    border-color: var(--purple) !important;
    background: var(--purple-dim) !important;
}
section[data-testid="stFileUploader"] button {
    background: var(--purple) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--r) !important;
    font-family: var(--font-head) !important;
    font-weight: 500 !important;
    font-size: .8rem !important;
    padding: 6px 16px !important;
    transition: background .15s;
}
section[data-testid="stFileUploader"] button:hover { background: var(--purple-lo) !important; }

/* ── Images ── */
.stImage > div {
    border-radius: var(--r) !important;
    overflow: hidden;
    border: 1px solid var(--line);
    background: var(--surface);
}
.stImage img { border-radius: var(--r) !important; }

/* ── Progress bar ── */
.stProgress > div {
    background: var(--line) !important;
    border-radius: 4px !important;
    height: 8px !important;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--purple), var(--orange)) !important;
    border-radius: 4px !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    height: 1px !important;
    background: var(--line) !important;
    margin: 1.5rem 0 !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: var(--r) !important;
    border: 1px solid var(--line) !important;
    background: var(--surface) !important;
    color: var(--txt) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--txt-3); border-radius: 3px; }

/* ── COMPONENTS ── */

/* Hero */
.hero { padding: 1rem 0 1.5rem 0; margin-bottom: 1rem; border-bottom: 1px solid var(--line); }
.hero h1 {
    margin: 0 0 .25rem 0 !important;
    font-size: 2.2rem !important;
    background: linear-gradient(90deg, var(--purple), var(--orange));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent;
}
.hero-sub { color: var(--txt-2) !important; font-size: .95rem; font-weight: 400; line-height: 1.5; }
.hero-meta {
    display: inline-block;
    margin-top: .75rem;
    padding: 4px 12px;
    border: 1px solid var(--purple-line);
    border-radius: var(--r);
    font-family: var(--font-mono);
    font-size: .75rem;
    color: var(--purple);
    background: var(--purple-dim);
    letter-spacing: .02em;
}

/* Verdict */
.verdict {
    font-family: var(--font-head);
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: .2rem 0 .1rem 0;
}
.verdict-yes { color: var(--ok); }
.verdict-no  { color: var(--no); }

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 4px 12px;
    border: 1px solid var(--purple-line);
    border-radius: var(--r);
    font-family: var(--font-mono);
    font-size: .9rem;
    font-weight: 600;
    color: var(--purple);
    background: var(--purple-dim);
}

.sub { color: var(--txt-3) !important; font-size: .85rem; line-height: 1.5; }

/* Latency table */
.lat-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .85rem;
    font-family: var(--font-mono);
    border: 1px solid var(--line);
    border-radius: var(--r);
    overflow: hidden;
    background: var(--card);
}
.lat-table th {
    background: var(--surface);
    color: var(--txt-3);
    font-family: var(--font-head);
    font-weight: 600;
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--line);
}
.lat-table th:last-child { text-align: right; }
.lat-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--line);
    color: var(--txt-2);
    font-variant-numeric: tabular-nums;
}
.lat-table td:last-child { text-align: right; color: var(--purple); font-weight: 600; }
.lat-table tr:last-child td { border-bottom: none; }
.lat-table tr.total-row td {
    font-weight: 700;
    color: var(--txt);
    border-top: 2px solid var(--purple-line);
    background: var(--purple-dim);
    padding: 12px 14px;
}
.lat-table tr.total-row td:last-child { color: var(--purple); font-size: .95rem; }
.lat-table td.off { opacity: .4; font-style: italic; color: var(--txt-3); }
.lat-table .step-note {
    display: block;
    font-size: .7rem;
    color: var(--txt-3);
    font-family: var(--font-body);
    margin-top: 2px;
}

/* Top-5 cards */
.top5-card {
    text-align: center;
    padding: .75rem .5rem;
    border: 1px solid var(--line);
    border-radius: var(--r);
    background: var(--card);
    transition: all .2s;
}
.top5-card:hover { border-color: var(--purple); box-shadow: 0 4px 12px rgba(124, 58, 237, 0.08); }
.top5-rank {
    font-family: var(--font-head);
    font-size: .65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--orange);
    margin-bottom: 4px;
}
.top5-score { font-family: var(--font-mono); font-size: .9rem; font-weight: 600; color: var(--purple); }
.top5-name { font-size: .75rem; color: var(--txt-3); margin-top: 4px; }

/* Sidebar footer */
.sidebar-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    font-size: .8rem;
    color: var(--txt-3);
    line-height: 1.6;
}
.sidebar-footer .model-name {
    font-family: var(--font-head);
    font-weight: 700;
    font-size: .85rem;
    letter-spacing: .02em;
    color: var(--purple);
}

/* Gradient divider */
.stripe-rule {
    height: 2px;
    background: linear-gradient(90deg, var(--purple) 0%, var(--orange) 100%);
    opacity: .25;
    margin: 1.5rem 0;
    border: none;
    border-radius: 1px;
}

/* Section label */
.section-label {
    font-family: var(--font-head);
    font-size: .75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--txt-3);
    margin-bottom: .75rem;
}

/* Setup steps */
.step-head {
    font-family: var(--font-head);
    font-weight: 600;
    font-size: 1rem;
    color: var(--txt);
    margin: 1.5rem 0 .25rem 0;
}
.step-head .step-num {
    display: inline-block;
    width: 22px; height: 22px;
    line-height: 22px;
    text-align: center;
    margin-right: 8px;
    border-radius: 50%;
    background: var(--purple);
    color: #fff;
    font-size: .72rem;
    font-weight: 700;
}
.fact {
    font-family: var(--font-mono);
    font-size: .8rem;
    color: var(--txt-2);
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 3px solid var(--purple);
    border-radius: var(--r);
    padding: 10px 14px;
    margin: .5rem 0;
    line-height: 1.7;
    word-break: break-all;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False, max_entries=2)
def get_embedder(weights_path, backbone, device, weights_id):
    """One loaded model per (checkpoint, backbone, device).

    `weights_id` is part of the key only so that replacing the file on disk
    invalidates the cache; it is not used inside.
    """
    return E.Embedder(weights_path, device, backbone).warmup()


@st.cache_resource(show_spinner="Loading enrolled gallery...")
def get_gallery(path, mtime):
    return P.load_gallery(path)


@st.cache_data(show_spinner=False)
def get_environment():
    return E.describe_environment()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

MISSING_THUMB = np.full((112, 112, 3), 235, dtype=np.uint8)


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_upload(f):
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error(f"Could not decode **{f.name}** - not a readable image file.")
        st.stop()
    return img


def gallery_thumb(gallery, index):
    """Enrolled image, or a neutral placeholder when the dataset has moved."""
    img = P.read_gallery_image(gallery, index)
    return rgb(img) if img is not None else MISSING_THUMB


def guarded_embed(embedder, arr):
    """Embed, turning a device failure into a readable message instead of a stack."""
    try:
        return embedder.embed(arr)
    except Exception as exc:
        st.error(
            f"Inference failed on **{embedder.device.upper()}**: "
            f"`{type(exc).__name__}`. If this is the GPU, it may be out of memory "
            "or the driver may have reset - switch to **CPU** in the sidebar and retry."
        )
        st.caption(str(exc)[:300])
        st.stop()


def guarded_align(bgr, ctx_id=0):
    """Detect + align, turning a missing face or detector failure into a
    readable message instead of a stack trace or a silently mis-aligned crop."""
    try:
        aligned, _bbox, _kps = P.detect_and_align(bgr, ctx_id=ctx_id)
        return aligned
    except P.NoFaceDetected:
        st.error("No face detected in that image - try a clearer, front-facing photo.")
        st.stop()
    except Exception as exc:
        st.error(f"Face detection failed: `{type(exc).__name__}`: {exc}")
        st.caption(str(exc)[:300])
        st.stop()


def latency_table(rows, total):
    html = [
        '<table class="lat-table">',
        '<tr><th>Pipeline Step</th><th>Latency</th></tr>',
    ]
    for label, ms, note in rows:
        note_html = f'<span class="step-note">{note}</span>' if note else ""
        if ms is None:
            html.append(f'<tr><td class="off">{label}{note_html}</td>'
                        f'<td class="off">excluded</td></tr>')
        else:
            html.append(f'<tr><td>{label}{note_html}</td><td>{ms:.2f} ms</td></tr>')
    html.append(f'<tr class="total-row"><td>Total</td><td>{total:.2f} ms</td></tr>')
    html.append("</table>")
    st.markdown("".join(html), unsafe_allow_html=True)


def stripe_rule():
    st.markdown('<div class="stripe-rule"></div>', unsafe_allow_html=True)


def step_head(num, text):
    st.markdown(f'<div class="step-head"><span class="step-num">{num}</span>{text}</div>',
                unsafe_allow_html=True)


def fact(text):
    st.markdown(f'<div class="fact">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SETUP SCREEN - shown on first run and whenever the model or dataset changes
# ─────────────────────────────────────────────────────────────────────────────

def setup_screen(settings, devices):
    st.markdown(
        """
        <div class="hero">
            <h1>Set up the experiment</h1>
            <p class="hero-sub">
                Pick the model checkpoint and the dataset. The 1:N gallery is
                enrolled once from that dataset, then the 1:1 and 1:N tests open.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    env = get_environment()
    if "gpu" in devices:
        st.success(f"PyTorch {env['torch']} - GPU available: {env['gpu_name']}")
    else:
        st.warning(
            f"PyTorch {env['torch']} - **no GPU available, CPU only**. "
            + (" ".join(env["notes"]) or "")
            + "  Run `python check_devices.py` for details."
        )

    # ---------------------------------------------------------------- weights
    step_head(1, "Model weights (.pt)")

    found = C.find_weights()
    labels = [f"{os.path.basename(p)}  -  {os.path.getsize(p) / 1e6:.0f} MB" for p in found]
    OTHER = "Type a path myself..."
    default_idx = 0
    if settings.get("weights_path") in found:
        default_idx = found.index(settings["weights_path"])

    choice = st.selectbox(
        f"Checkpoints found in `{C.MODELS_DIR}`" if found
        else "No checkpoint found in the models folder",
        labels + [OTHER],
        index=default_idx if found else len(labels),
    )
    if choice == OTHER:
        weights_path = st.text_input(
            "Full path to the .pt file",
            value=settings.get("weights_path", ""),
            placeholder=r"C:\Users\lenovo\Downloads\app_4 (1)\app_4\models\Glint360K_R100_TopoFR_9760.pt",
        ).strip('"').strip()
    else:
        weights_path = found[labels.index(choice)]

    backbone = st.selectbox(
        "Backbone", ["auto-detect"] + list(E.BACKBONES),
        help="Auto-detect reads the block counts out of the checkpoint. "
             "Override only if detection fails.",
    )
    backbone = None if backbone == "auto-detect" else backbone

    if weights_path and os.path.exists(weights_path):
        if st.button("Inspect checkpoint"):
            with st.spinner("Reading checkpoint..."):
                try:
                    st.session_state.inspection = E.inspect_weights(weights_path)
                except Exception as exc:
                    st.session_state.inspection = None
                    st.error(f"Could not read the checkpoint: {exc}")
        info = st.session_state.get("inspection")
        if info and info["path"] == weights_path:
            line = (f"{info['name']} &middot; {info['size_mb']:.0f} MB<br>"
                    f"backbone <b>{info['backbone'] or 'unknown'}</b> &middot; "
                    f"{info['embedding_dim']}-d embeddings")
            if info["train_identities"]:
                line += f" &middot; trained on {info['train_identities']:,} identities"
            fact(line)
    elif weights_path:
        st.error(f"File not found: `{weights_path}`")

    # ---------------------------------------------------------------- dataset
    step_head(2, "Dataset")
    st.markdown(
        '<p class="sub">One folder per identity, each containing a '
        '<code>reference</code> and a <code>match</code> image. A <code>.zip</code> '
        'of that folder works too.</p>',
        unsafe_allow_html=True,
    )

    known = C.find_datasets()
    source = st.radio(
        "Source", ["Folder on this machine", "Zip archive"],
        horizontal=True, label_visibility="collapsed",
    )

    dataset_root = ""
    if source == "Folder on this machine":
        if known:
            names = [os.path.basename(p) for p in known]
            previous = settings.get("dataset_root")
            index = known.index(previous) if previous in known else 0
            picked = st.selectbox(f"Folders found in `{C.DATA_DIR}`",
                                  names + [OTHER], index=index)
            dataset_root = "" if picked == OTHER else known[names.index(picked)]
        if not dataset_root:
            dataset_root = st.text_input(
                "Full path to the dataset folder",
                value=settings.get("dataset_root", ""),
                placeholder=r"C:\path\to\final_dataset",
            ).strip('"').strip()
    else:
        zip_path = st.text_input(
            "Full path to the .zip", placeholder=r"C:\path\to\final_dataset.zip"
        ).strip('"').strip()
        up = st.file_uploader("...or upload it", type=["zip"])
        if up is not None:
            os.makedirs(C.EXTRACT_DIR, exist_ok=True)
            tmp = os.path.join(C.EXTRACT_DIR, up.name)
            with open(tmp, "wb") as fh:
                fh.write(up.getbuffer())
            zip_path = tmp
        if zip_path and st.button("Extract archive"):
            if not os.path.exists(zip_path):
                st.error(f"File not found: `{zip_path}`")
            else:
                with st.spinner("Extracting..."):
                    try:
                        st.session_state.extracted = D.extract_zip(zip_path, C.EXTRACT_DIR)
                    except Exception as exc:
                        st.error(f"Could not extract: {exc}")
        dataset_root = st.session_state.get("extracted", "")
        if dataset_root:
            fact(f"extracted to<br>{dataset_root}")

    if dataset_root and st.button("Scan dataset"):
        with st.spinner("Scanning folders..."):
            # Keyed by the folder asked for, so a stale scan is never shown after
            # the user edits the path.
            st.session_state.scan = (dataset_root, D.scan(dataset_root))

    scanned_for, scan = st.session_state.get("scan", (None, None))
    if scan is not None and scanned_for != dataset_root:
        scan = None
        st.info("Path changed - scan it again.")
    if scan:
        for w in scan["warnings"]:
            st.warning(w)
        if scan["identities"]:
            fact(f"{D.summary(scan)}<br>root: {scan['root']}")

    # ---------------------------------------------------------------- enroll
    ready = bool(scan and scan["identities"] and weights_path
                 and os.path.exists(weights_path))
    step_head(3, "Build the 1:N gallery")

    if not ready:
        st.info("Pick a checkpoint, then scan a dataset, to enable enrollment.")
        return

    total_ids = len(scan["identities"])
    col_a, col_b, col_c = st.columns(3)
    n_enroll = col_a.number_input(
        "Identities to enroll", min_value=0, max_value=total_ids, value=total_ids,
        step=100, help="0 enrolls everything.",
    )
    n_enroll = total_ids if n_enroll == 0 else int(n_enroll)
    enroll_device = col_b.radio(
        "Enroll on", devices, horizontal=True,
        index=len(devices) - 1,            # prefer the GPU: enrollment is much faster
        format_func=lambda d: d.upper(),
    )
    batch = col_c.number_input("Batch size", min_value=1, max_value=256, value=32, step=8)

    weights_id = E.weights_fingerprint(weights_path)
    ds_id = D.dataset_id(scan["root"])
    out_path = C.gallery_path(weights_id, ds_id, n_enroll)

    if os.path.exists(out_path):
        st.info("A gallery already exists for this model, dataset and size.")
        if st.button("Use the existing gallery", type="primary"):
            _finish(settings, weights_path, backbone, weights_id, scan["root"], out_path)

    if st.button(f"Enroll {n_enroll:,} identities and start", type="primary"):
        try:
            with st.spinner(f"Loading {os.path.basename(weights_path)} on "
                            f"{enroll_device.upper()}..."):
                embedder = get_embedder(weights_path, backbone, enroll_device, weights_id)
        except Exception as exc:
            st.error(f"Could not load the model: {exc}")
            st.stop()

        bar = st.progress(0.0)
        status = st.empty()

        def on_progress(done, total, rate):
            bar.progress(done / max(total, 1))
            eta = (total - done) / max(rate, 1e-6)
            status.markdown(
                f'<p class="sub">{done:,} / {total:,} &nbsp;&middot;&nbsp; '
                f'{rate:.1f} img/s &nbsp;&middot;&nbsp; eta {eta:.0f}s</p>',
                unsafe_allow_html=True)

        try:
            path, meta = build_gallery(
                embedder, scan["identities"][:n_enroll], scan["root"], out_path,
                batch=int(batch), progress=on_progress)
        except Exception as exc:
            st.error(f"Enrollment failed: {type(exc).__name__}: {exc}")
            st.stop()

        bar.progress(1.0)
        st.success(f"Enrolled {meta['n_identities']:,} identities in "
                   f"{meta['elapsed_s']:.0f}s on {enroll_device.upper()}.")
        _finish(settings, weights_path, backbone, weights_id, scan["root"], path)


def _finish(settings, weights_path, backbone, weights_id, root, gallery_path):
    """Persist the choice and drop into the test screens."""
    settings.update({
        "weights_path": weights_path,
        "backbone": backbone,
        "weights_id": weights_id,
        "dataset_root": root,
        "gallery_path": gallery_path,
    })
    C.save_settings(settings)
    st.session_state.setup_open = False
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────────────

settings = C.load_settings()
devices = E.available_devices()

configured = (
    settings.get("weights_path") and os.path.exists(settings["weights_path"])
    and settings.get("gallery_path") and os.path.exists(settings["gallery_path"])
)
if "setup_open" not in st.session_state:
    st.session_state.setup_open = not configured

if st.session_state.setup_open or not configured:
    with st.sidebar:
        st.markdown("### Setup")
        st.caption("Choose a checkpoint and a dataset to begin.")
        if configured and st.button("Cancel"):
            st.session_state.setup_open = False
            st.rerun()
    setup_screen(settings, devices)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

gallery = get_gallery(settings["gallery_path"],
                      os.path.getmtime(settings["gallery_path"]))
if gallery is None:
    st.error(f"Could not read the gallery at `{settings['gallery_path']}`. "
             "Rebuild it from the setup screen.")
    if st.button("Open setup"):
        st.session_state.setup_open = True
        st.rerun()
    st.stop()

with st.sidebar:
    st.markdown("### Configuration")

    device_labels = {d: E.device_label(d) for d in devices}
    device = st.radio(
        "Inference device",
        devices,
        format_func=lambda d: device_labels.get(d, d),
        help="The same PyTorch model and the same checkpoint on both devices - "
             "only the device changes.",
    )

    if device == "gpu":
        st.success(f"PyTorch on {E.gpu_backend().upper()}")
    else:
        st.caption("Running on `torch.device('cpu')`.")

    st.markdown("---")
    threshold = st.slider(
        "Decision threshold (cosine)",
        0.0, 1.0, C.VERIF_THRESHOLD, 0.005,
        help="Default 0.225 - the calibrated operating point for the Glint360K "
             "TopoFR-R100 checkpoint. Every model has its own threshold, so "
             "recalibrate before trusting this value on a different checkpoint.",
    )

    st.markdown("---")
    if st.button("Change model / dataset"):
        st.session_state.setup_open = True
        st.session_state.pop("scan", None)
        st.rerun()

    meta = gallery["meta"] if gallery else {}
    st.markdown(
        f'<div class="sidebar-footer">'
        f'<span class="model-name">TopoFR-{(meta.get("backbone") or "?").upper()}</span><br>'
        f'{meta.get("weights_name", "?")}<br>'
        f'512-d &middot; PyTorch<br>'
        f'Gallery: <code>{len(gallery["names"]):,}</code> identities, pre-embedded'
        f'</div>',
        unsafe_allow_html=True,
    )

if "gpu" not in devices:
    st.sidebar.info("No PyTorch GPU on this machine - CPU only. "
                    "See `check_devices.py`.")

# The gallery is only meaningful for the model that produced it.
if gallery and not P.gallery_matches(gallery, settings.get("weights_id")):
    st.error(
        f"This gallery was enrolled with **{meta.get('weights_name', 'another model')}**, "
        "which is not the checkpoint currently selected. Embeddings from two models "
        "are not comparable - rebuild the gallery from **Change model / dataset**."
    )
    st.stop()

try:
    embedder = get_embedder(settings["weights_path"], settings.get("backbone"),
                            device, settings["weights_id"])
except Exception as exc:
    st.error(f"Could not load the model on {device.upper()}: {exc}")
    st.stop()

sync = embedder.sync


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="hero">
        <h1>TopoFR-{(meta.get("backbone") or "?").upper()}</h1>
        <p class="hero-sub">
            Per-step latency &middot; batch-1 inference &middot;
            RetinaFace detect + align &rarr; 112&times;112 crop
        </p>
        <span class="hero-meta">
            PyTorch &nbsp;/&nbsp; {device_labels.get(device, device)}
            &nbsp;/&nbsp; 512-d embeddings
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_11, tab_1n = st.tabs(["1:1  Verification", "1:N  Identification"])


# ─────────────────────────────────────────────────────────────────────────────
# 1:1 VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

with tab_11:
    st.markdown(
        "Drop two face images. Each is embedded to **512 dimensions** and scored "
        "by **cosine similarity**."
    )

    c1, c2 = st.columns(2)
    up_a = c1.file_uploader("Image A", type=["jpg", "jpeg", "png", "bmp"], key="a")
    up_b = c2.file_uploader("Image B", type=["jpg", "jpeg", "png", "bmp"], key="b")

    if up_a and up_b:
        img_a, img_b = read_upload(up_a), read_upload(up_b)
        ctx_id = 0 if device == "gpu" else -1

        aligned_a, t_det_a = P.timed(lambda: guarded_align(img_a, ctx_id))
        aligned_b, t_det_b = P.timed(lambda: guarded_align(img_b, ctx_id))

        arr_a, t_pre_a = P.timed(lambda: P.to_tensor(aligned_a))
        emb_a, t_emb_a = P.timed(lambda: guarded_embed(embedder, arr_a), sync)
        arr_b, t_pre_b = P.timed(lambda: P.to_tensor(aligned_b))
        emb_b, t_emb_b = P.timed(lambda: guarded_embed(embedder, arr_b), sync)

        (na, nb), t_norm = P.timed(lambda: (P.normalize(emb_a), P.normalize(emb_b)))
        score, t_cmp = P.timed(lambda: P.cosine(na, nb))
        same, t_dec = P.timed(lambda: bool(score >= threshold))

        stripe_rule()
        left, mid, right = st.columns([1, 1, 1.15])

        with left:
            st.markdown('<div class="section-label">Image A</div>', unsafe_allow_html=True)
            st.image(rgb(img_a), use_container_width=True)
            #st.caption("Aligned crop fed to the model")
            #st.image(rgb(aligned_a), width=112)
        with mid:
            st.markdown('<div class="section-label">Image B</div>', unsafe_allow_html=True)
            st.image(rgb(img_b), use_container_width=True)
            #st.caption("Aligned crop fed to the model")
            #st.image(rgb(aligned_b), width=112)
        with right:
            st.markdown('<div class="section-label">Decision</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="verdict {"verdict-yes" if same else "verdict-no"}">'
                f'{"Same person" if same else "Different people"}</div>'
                f'<span class="score-badge">cosine {score:.4f}</span>'
                f'<p class="sub" style="margin-top:.6rem">threshold {threshold:.3f}</p>',
                unsafe_allow_html=True,
            )

        stripe_rule()
        st.markdown(
            f'<div class="section-label">Latency - {device_labels.get(device, device)}</div>',
            unsafe_allow_html=True,
        )
        total = (t_det_a + t_det_b + t_pre_a + t_emb_a + t_pre_b + t_emb_b
                 + t_norm + t_cmp + t_dec)
        latency_table([
            ("Detect + align A", t_det_a, "RetinaFace, 5-pt affine -> 112x112"),
            ("Detect + align B", t_det_b, "RetinaFace, 5-pt affine -> 112x112"),
            ("Preprocess A", t_pre_a, "resize, RGB, [-1,1]"),
            ("Embed A", t_emb_a, ""),
            ("Preprocess B", t_pre_b, "resize, RGB, [-1,1]"),
            ("Embed B", t_emb_b, ""),
            ("Normalise", t_norm, "L2 unit vectors"),
            ("Compare", t_cmp, "cosine similarity"),
            ("Decide", t_dec, f"score >= {threshold:.3f}"),
        ], total)
    else:
        st.info("Waiting for two images.")


# ─────────────────────────────────────────────────────────────────────────────
# 1:N IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

with tab_1n:
    G = gallery["emb"]
    n_gal = len(gallery["names"])
    st.markdown(
        f"Drop one image. It is embedded and matched against **{n_gal:,} enrolled "
        "identities**. The gallery was embedded and normalised at enrollment, so "
        "only the probe's cost is timed."
    )

    up_p = st.file_uploader("Probe image", type=["jpg", "jpeg", "png", "bmp"], key="p")

    if up_p:
        img_p = read_upload(up_p)
        ctx_id = 0 if device == "gpu" else -1

        aligned_p, t_det = P.timed(lambda: guarded_align(img_p, ctx_id))
        arr_p, t_pre = P.timed(lambda: P.to_tensor(aligned_p))
        emb_p, t_emb = P.timed(lambda: guarded_embed(embedder, arr_p), sync)
        npb, t_norm = P.timed(lambda: P.normalize(emb_p))
        sims, t_cmp = P.timed(lambda: (npb @ G.T).ravel())
        top, t_dec = P.timed(lambda: np.argsort(-sims)[:5])

        scores = sims[top]
        accepted = scores[0] >= threshold

        left, mid, right = st.columns([1, 1, 1.15])

        with left:
            st.markdown('<div class="section-label">Probe</div>', unsafe_allow_html=True)
            st.image(rgb(img_p), use_container_width=True)
            st.caption("Aligned crop fed to the model")
            st.image(rgb(aligned_p), width=112)

        with mid:
            st.markdown('<div class="section-label">Best Match</div>', unsafe_allow_html=True)
            st.image(gallery_thumb(gallery, top[0]), use_container_width=True)
            label = (f"identity {gallery['names'][top[0]]}" if accepted
                     else "No match above threshold")
            st.markdown(
                f'<div class="verdict {"verdict-yes" if accepted else "verdict-no"}" '
                f'style="font-size:1.15rem">{label}</div>'
                f'<span class="score-badge">{scores[0]:.4f}</span>',
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                f'<div class="section-label">Latency - {device_labels.get(device, device)}</div>',
                unsafe_allow_html=True,
            )
            total = t_det + t_pre + t_emb + t_norm + t_cmp + t_dec
            latency_table([
                ("Detect + align probe", t_det, "RetinaFace, 5-pt affine -> 112x112"),
                ("Preprocess probe", t_pre, "resize, RGB, [-1,1]"),
                ("Embed probe", t_emb, ""),
                ("Normalise probe", t_norm, "L2 unit vector"),
                ("Search gallery", t_cmp, f"1 x {n_gal:,} cosine"),
                ("Rank / decide", t_dec, "top-5 argsort"),
                ("Embed gallery", None, f"{n_gal:,} identities, done at enrollment"),
            ], total)

        stripe_rule()
        st.markdown('<div class="section-label">Top 5</div>', unsafe_allow_html=True)

        cols = st.columns(5)
        for idx, (col, i, s) in enumerate(zip(cols, top, scores)):
            rank_label = ["1st", "2nd", "3rd", "4th", "5th"][idx]
            col.image(gallery_thumb(gallery, i), use_container_width=True)
            col.markdown(
                f'<div class="top5-card">'
                f'<div class="top5-rank">{rank_label}</div>'
                f'<div class="top5-score">{s:.4f}</div>'
                f'<div class="top5-name">id {gallery["names"][i]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Waiting for a probe image.")
