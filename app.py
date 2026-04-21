"""Streamlit web interface for Melody Matcher."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.search.search_service import search_fuzzy_match, search_exact_match
from melody_matcher.storage.index_store import load_index

_DEFAULT_INDEX = _REPO_ROOT / "data" / "index" / "melody_index.json"
_NOTE_WITHOUT_OCTAVE = re.compile(r"^([A-Ga-g][#b]?)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_notes(raw: str) -> list[str]:
    return [
        token + "4" if _NOTE_WITHOUT_OCTAVE.match(token) else token
        for token in raw.split()
    ]


@st.cache_data(show_spinner="Loading melody index…")
def _load_index(path: str) -> dict:
    return load_index(path)


def _score_colour(score: float) -> str:
    if score >= 0.9:
        return "#2ecc71"   # green
    if score >= 0.7:
        return "#f39c12"   # amber
    return "#e74c3c"       # red


def _confidence_badge(score: float) -> str:
    colour = _score_colour(score)
    label = "Exact match" if score == 1.0 else f"{score * 100:.1f}% match"
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.78rem;font-weight:600">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Melody Matcher",
    page_icon="🎵",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Search settings")
    min_score_pct = st.slider(
        "Minimum similarity",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        format="%d%%",
        help="Discard results below this confidence threshold.",
    )
    max_results = st.number_input(
        "Max results",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Maximum number of ranked results to display.",
    )
    st.divider()
    st.caption(
        "Notes are matched by their **interval signature** — the pattern of "
        "semitone jumps between consecutive notes — so the search is "
        "transposition-invariant."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎵 Melody Matcher")
st.subheader("Find songs by humming a few notes")
st.markdown(
    "Type a sequence of notes (e.g. **`C D E F G`** or **`C4 D4 E4`**).  \n"
    "Octave 4 is assumed when no octave number is given.  \n"
    "Use sharps as `F#` and flats as `Bb`."
)
st.divider()

# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

index = _load_index(str(_DEFAULT_INDEX))

if not index:
    st.warning(
        "⚠️  No index found.  "
        "Run `python scripts/build_index.py` first to index your MIDI files.",
        icon="⚠️",
    )
    st.stop()

sig_count = len(index)
entry_count = sum(len(v) for v in index.values())
st.caption(f"Index loaded · {sig_count:,} signatures · {entry_count:,} segments")

# ---------------------------------------------------------------------------
# Search input
# ---------------------------------------------------------------------------

col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")

with col_input:
    raw_query = st.text_input(
        "Enter notes",
        placeholder="e.g.  C  D  E  F  G  A  G",
        label_visibility="collapsed",
    )

with col_btn:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------

if search_clicked or raw_query:
    if not raw_query.strip():
        st.info("Type at least two notes above and press Search.")
        st.stop()

    notes = _normalise_notes(raw_query.strip())

    if len(notes) < 2:
        st.warning("Please enter **at least 2 notes** to search.")
        st.stop()

    # Show what we actually searched
    try:
        from melody_matcher.features.interval_encoder import encode_intervals
        sig = encode_intervals(notes)
        st.markdown(
            f"**Query:** `{' '.join(notes)}`  →  "
            f"interval signature `{sig}`"
        )
    except ValueError as exc:
        st.error(f"Could not encode notes: {exc}")
        st.stop()

    min_score = min_score_pct / 100.0

    try:
        exact_hits = {
            (h["file"], h["segment_index"])
            for h in search_exact_match(notes, index)
        }
        results = search_fuzzy_match(
            notes,
            index,
            top_n=int(max_results),
            min_score=min_score,
        )
    except ValueError as exc:
        st.error(f"Search error: {exc}")
        st.stop()

    # Promote exact hits to 1.0 in the result list
    for r in results:
        if (r["file"], r["segment_index"]) in exact_hits:
            r["score"] = 1.0
    results.sort(key=lambda r: r["score"], reverse=True)

    st.divider()

    if not results:
        st.info("🔍  No matches found above the similarity threshold.  Try lowering it in the sidebar.")
        st.stop()

    st.markdown(f"### Results &nbsp; <small style='color:grey'>{len(results)} match(es)</small>", unsafe_allow_html=True)

    # --- Metric summary row ---
    best = results[0]
    exact_count = sum(1 for r in results if r["score"] == 1.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Results found", len(results))
    c2.metric("Exact matches", exact_count)
    c3.metric("Best score", f"{best['score'] * 100:.1f}%")

    st.divider()

    # --- Result cards ---
    for rank, hit in enumerate(results, start=1):
        score = hit["score"]
        file_name = Path(hit["file"]).name
        file_path = hit["file"]
        seg_idx = hit["segment_index"]

        with st.container():
            left, right = st.columns([6, 2])
            with left:
                st.markdown(
                    f"**{rank}.&ensp;{file_name}**  \n"
                    f"<span style='color:grey;font-size:0.82rem'>{file_path} &nbsp;·&nbsp; segment #{seg_idx}</span>",
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown(
                    f"<div style='text-align:right;padding-top:6px'>"
                    f"{_confidence_badge(score)}</div>",
                    unsafe_allow_html=True,
                )
            # Thin progress bar as visual score indicator
            st.progress(score)
            st.write("")  # breathing room between cards