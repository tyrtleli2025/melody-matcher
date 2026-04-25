# CLAUDE.md — Melody Matcher Project Guide

## Project Overview

**Melody Matcher** is a query-by-humming (QbH) + query-by-MIDI search engine built on the
**Master Blueprint v2: Multimodal ML Melody Search** architecture.

A user hums, sings, or uploads a MIDI clip. The system:
1. Transcribes audio to a symbolic (MIDI) representation via **Basic Pitch** (Spotify, Apache-2.0)
2. Embeds the melody into a shared vector space using pre-trained backbones (**MERT** for audio, **MidiBERT/MusicBERT** for symbolic)
3. Retrieves nearest neighbors from a pre-indexed corpus via **FAISS**
4. Reranks results using melodic similarity + popularity signals from **Last.fm** (Spotify Web API is deprecated — do not use it)

Melodies are represented **key- and tempo-invariantly** from the start: pitch intervals (semitone deltas) and normalized inter-onset intervals (IOIs), not absolute MIDI pitch or absolute timing.

---

## Roadmap (Master Blueprint v2 — 8 Phases)

| Phase | Name | Description | Status |
|-------|------|-------------|--------|
| **0** | Problem Framing & Evaluation Harness | Define query distribution, build frozen eval sets (MIR-QBSH, Humtrans, hand-recorded), define metrics (MRR@10, Hit@k), stand up `eval.py` leaderboard script | 🔄 **IN PROGRESS** |
| **1** | Data Sourcing & Preprocessing | Lakh MIDI Dataset (LMD-matched ~45k), melody extraction (skyline + track heuristics), normalization to intervals+IOI+Parsons code, metadata via MusicBrainz, popularity via Last.fm, deduplication, output `songs.parquet` | ⬜ TODO |
| **2** | Baseline Retrieval System | LSH index over n-gram shingles (n=5,8,12), DTW reranking on top-200 candidates, combined melodic+popularity score. Must ship before any neural work | ⬜ TODO |
| **3** | Neural Embedding Model | MERT (audio) + MidiBERT (symbolic) backbones, cross-modal projection heads (InfoNCE loss, 256-dim shared space). Fine-tune only the projection heads — do NOT train from scratch | ⬜ TODO |
| **4** | Audio-to-Symbolic Pipeline | Basic Pitch for audio→MIDI, wrapped with Silero VAD + loudness normalization pre-processing and monophonic forcing + note merging post-processing | ⬜ TODO |
| **5** | Vector Database & Retrieval | FAISS `IndexHNSWFlat` (M=32, efConstruction=200), windowed indexing (8s windows, 16s stride), metadata in SQLite/Parquet keyed by FAISS ID | ⬜ TODO |
| **6** | Ranking | Scoring: `w_sim * cosine_sim + w_pop * popularity_log + w_conf * transcription_confidence`. Diversity filter (deduplicate by artist in top 10). "Melodic cluster" deep-dive view | ⬜ TODO |
| **7** | Frontend / UX | Streamlit prototype → Next.js + FastAPI production. Mic recording, file upload, piano roll input. Piano roll transcription preview before search. Popularity slider | ⬜ TODO |
| **8** | Evaluation, Iteration & Deployment | CI eval on every retrieval commit, online logging, A/B testing. FastAPI on GPU instance (T4), FAISS in-memory, Basic Pitch in-browser (WASM TypeScript port) | ⬜ TODO |

---

## Current Status

> **Phase 0.2 — Building the Evaluation Harness**

- [ ] 0.1 Define query distribution (noisy-snippet quadrant is the priority target)
- [x] 0.2 Build eval sets: MIR-QBSH corpus, Humtrans, hand-recorded set — **IN PROGRESS**
- [ ] 0.3 Define metrics: MRR@10 (primary), Hit@{1,5,10}, latency p50/p95, popularity-weighted Hit@10
- [ ] 0.4 Stand up `scripts/v2/eval.py` leaderboard script (frozen — never train on these sets)

**Target metrics (v1 system):**
| Metric | Target |
|--------|--------|
| Hit@1 on hummed queries (clean) | ≥ 50% |
| Hit@10 on hummed queries (clean) | ≥ 80% |
| Hit@10 on hummed queries (phone mic, noisy) | ≥ 55% |
| MIDI-clip query Hit@1 (25-note excerpt) | ≥ 85% |
| End-to-end latency p95 (hum → results) | < 2.5 s |
| Corpus size | 100k–170k unique melodies |

---

## Technical Stack (source of truth: Master Blueprint v2 PDF)

| Layer | Choice | Notes |
|-------|--------|-------|
| MIDI parsing | `pretty_midi`, `mido` | Standard, maintained |
| Key detection | `music21` (Krumhansl-Schmuckler) | Good enough for pop |
| Audio I/O | `librosa`, `soundfile` | Standard |
| VAD | Silero VAD | Small, accurate |
| Audio → MIDI | `basic-pitch` (Spotify, Apache-2.0) | SOTA for this task; do NOT use raw CREPE |
| Audio embedding | MERT-v1-95M (HuggingFace, MIT) | Pre-trained on 160k hours of audio |
| Symbolic embedding | MidiBERT / MusicBERT | Pre-trained on MIDI corpora |
| Vector DB | FAISS `IndexHNSWFlat` | Fastest, free, local; migrate to hosted only for multi-tenant |
| Popularity | Last.fm API + Spotify Charts CSVs | Spotify Web API deprecated Nov 2024 — do not use |
| Metadata | MusicBrainz | Canonical IDs |
| Backend | FastAPI | Async, typed |
| Frontend (prototype) | Streamlit | Fast iteration |
| Frontend (prod) | Next.js + `basic-pitch-ts` (WASM) | Browser-side transcription |
| Eval | Custom `eval.py`, MIR-QBSH, Humtrans | Must be frozen early |

---

## Style Guide

- **Python version:** 3.10+ with strict type hints on all function signatures
- **File operations:** Use `pathlib.Path` exclusively — no `os.path` string manipulation
- **Directory structure:** Follow the blueprint layout; scripts live in `scripts/v2/`, source modules in `src/`
- **Melody representation:** Always use pitch intervals + normalized IOIs as the canonical form; raw MIDI pitch/absolute timing are only intermediate representations
- **Popularity scores:** Store as normalized log-scores in [0, 1] — never raw counts
- **No Spotify Web API:** The audio-features, audio-analysis, recommendations, and related-artists endpoints are deprecated. Use Last.fm as the popularity workhorse
- **Eval sets are frozen:** Never train on MIR-QBSH test split or Humtrans test split

---

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run the evaluation leaderboard script (once created)
python scripts/v2/eval.py

# Build/rebuild the corpus index (once created)
python scripts/v2/build_index.py

# Run tests
pytest tests/
```

---

## Key Design Decisions (do not revisit without cause)

1. **Basic Pitch over raw CREPE** — Basic Pitch outputs MIDI directly with onset/offset detection. CREPE outputs a pitch curve that requires non-trivial post-processing.
2. **Baseline before neural** — The LSH+DTW baseline (Phase 2) must be live and evaluated before any Phase 3 neural work begins. Published 2005–2015 QbH baselines hit Hit@10 of 60–75% on MIR-QBSH; a neural system scoring below that is a regression.
3. **Pre-trained backbones only** — Fine-tune MERT+MidiBERT projection heads (~1M params), not a Transformer trained from scratch (~100M params). Train-from-scratch is Phase 8 experiment territory only.
4. **Windowed FAISS indexing** — Index 8-second windows (16s stride) per song, not one embedding per song. Users hum the hook; a whole-song embedding averages over irrelevant structure.
5. **InfoNCE over triplet loss** — InfoNCE with in-batch negatives gives N² contrasts for N forward passes. Use batch size ≥ 512.
6. **Popularity slider is real** — `w_pop` (default 0.20) is user-adjustable. Log every slider position; it is ground-truth preference data.
