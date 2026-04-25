"""Phase 0.4 — Evaluation leaderboard script.

Usage
-----
    source venv/bin/activate
    python scripts/v2/eval.py [--dataset {mir_qbsh,hand_recorded}] [--k K] [--ranker dummy]

This script is the single source of truth for offline evaluation.  Every
retrieval model must be plugged in here and improve the numbers; a PR that
drops MRR@10 by more than 2 points is blocked.

How to add a real ranker
------------------------
1. Implement a callable with signature::

       def my_ranker(query_audio_path: Path) -> list[str]:
           ...  # return song IDs ordered by descending relevance

2. Register it in ``_RANKERS`` at the bottom of this file.
3. Pass ``--ranker my_ranker`` on the command line.

Invariants
----------
* The eval sets in ``data/v2/eval/`` are FROZEN — never train on them.
* This script must run to completion even when the dataset directory is empty
  (it prints "0 queries evaluated" and exits cleanly with code 0).
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap — make the src package importable without `pip install -e .`
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from melody_matcher.eval.dataset_loader import EvalQuery, MIR_QBSH_Loader  # noqa: E402
from melody_matcher.eval.metrics import compute_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset roots
# ---------------------------------------------------------------------------
_EVAL_ROOT = _REPO_ROOT / "data" / "v2" / "eval"

_DATASET_ROOTS: dict[str, Path] = {
    "mir_qbsh": _EVAL_ROOT / "mir_qbsh",
    "hand_recorded": _EVAL_ROOT / "hand_recorded",
}

# Fake song IDs used by the dummy ranker to simulate a corpus.
_DUMMY_CORPUS: list[str] = [f"song_{i:04d}" for i in range(200)]


# ---------------------------------------------------------------------------
# Ranker implementations
# ---------------------------------------------------------------------------


def dummy_ranker(query_audio_path: Path) -> list[str]:  # noqa: ARG001
    """Return a random top-10 ranking from the dummy corpus.

    This ranker exists solely to let eval.py run end-to-end before any real
    retrieval model is wired in.  Expected MRR@10 ≈ 0.005 (random baseline).
    """
    return random.sample(_DUMMY_CORPUS, k=10)


# Map CLI name → callable.  Add real rankers here as they are implemented.
_RANKERS: dict[str, Callable[[Path], list[str]]] = {
    "dummy": dummy_ranker,
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_queries(dataset: str) -> list[EvalQuery]:
    """Load queries for the requested dataset.  Returns [] on empty dir."""
    root = _DATASET_ROOTS.get(dataset)
    if root is None:
        log.error("Unknown dataset %r.  Choose from: %s", dataset, list(_DATASET_ROOTS))
        sys.exit(1)

    if dataset == "hand_recorded":
        # hand_recorded uses the same two-level layout as MIR-QBSH.
        loader = MIR_QBSH_Loader(root)
    else:
        loader = MIR_QBSH_Loader(root)

    return loader.load_typed()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _run_eval(
    queries: list[EvalQuery],
    ranker: Callable[[Path], list[str]],
    k: int,
    popularity_scores: Optional[dict[str, float]],
) -> tuple[dict[str, float], float]:
    """Run the ranker over all queries and return (metrics_dict, elapsed_seconds)."""
    ranked_lists: list[list[str]] = []
    ground_truths: list[str] = []

    t_start = time.perf_counter()
    for query in queries:
        results = ranker(query.query_audio_path)
        ranked_lists.append(results)
        ground_truths.append(query.ground_truth_song_id)
    elapsed = time.perf_counter() - t_start

    metrics = compute_all(ranked_lists, ground_truths, popularity_scores=popularity_scores)
    return metrics, elapsed


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_COL_WIDTH = 28
_VAL_WIDTH = 8
_DIVIDER = "─" * (_COL_WIDTH + _VAL_WIDTH + 5)

_METRIC_LABELS: dict[str, str] = {
    "mrr@10": "MRR@10  (primary)",
    "hit@1": "Hit@1",
    "hit@5": "Hit@5",
    "hit@10": "Hit@10",
    "pop_weighted_hit@10": "Pop-weighted Hit@10",
}

_V1_TARGETS: dict[str, float] = {
    "hit@1": 0.50,
    "hit@10": 0.80,
    "mrr@10": 0.0,   # no explicit MRR target in blueprint — shown for info
}


def _format_report(
    ranker_name: str,
    dataset: str,
    n_queries: int,
    metrics: dict[str, float],
    elapsed: float,
    k: int,
) -> str:
    lines: list[str] = []

    lines.append("")
    lines.append("╔" + "═" * (_COL_WIDTH + _VAL_WIDTH + 3) + "╗")
    lines.append(f"║  Melody Matcher — Evaluation Leaderboard{' ' * (_COL_WIDTH + _VAL_WIDTH - 38)}║")
    lines.append("╠" + "═" * (_COL_WIDTH + _VAL_WIDTH + 3) + "╣")

    def row(label: str, value: str) -> str:
        return f"║  {label:<{_COL_WIDTH}}{value:>{_VAL_WIDTH}}  ║"

    lines.append(row("Dataset", dataset))
    lines.append(row("Ranker", ranker_name))
    lines.append(row("Queries evaluated", str(n_queries)))
    lines.append(row("Rank cutoff (k)", str(k)))
    lines.append(row("Total latency", f"{elapsed:.2f}s"))
    if n_queries > 0:
        lines.append(row("Avg latency / query", f"{elapsed / n_queries * 1000:.1f}ms"))

    lines.append("╠" + "═" * (_COL_WIDTH + _VAL_WIDTH + 3) + "╣")
    lines.append(f"║  {'Metric':<{_COL_WIDTH}}{'Score':>{_VAL_WIDTH}}  ║")
    lines.append("╠" + "─" * (_COL_WIDTH + _VAL_WIDTH + 3) + "╣")

    for key in ["mrr@10", "hit@1", "hit@5", "hit@10", "pop_weighted_hit@10"]:
        if key not in metrics:
            continue
        label = _METRIC_LABELS.get(key, key)
        score = metrics[key]
        value_str = f"{score:.4f}"

        target = _V1_TARGETS.get(key)
        if target and score >= target:
            flag = " ✓"
        elif target and score > 0:
            flag = " ·"
        else:
            flag = "  "

        lines.append(f"║  {label:<{_COL_WIDTH}}{value_str:>{_VAL_WIDTH}}{flag}║")

    lines.append("╠" + "─" * (_COL_WIDTH + _VAL_WIDTH + 3) + "╣")
    lines.append(f"║  {'v1 targets: Hit@1≥0.50  Hit@10≥0.80':<{_COL_WIDTH + _VAL_WIDTH + 1}}  ║")
    lines.append("╚" + "═" * (_COL_WIDTH + _VAL_WIDTH + 3) + "╝")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Melody Matcher offline evaluation leaderboard (Phase 0.4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=list(_DATASET_ROOTS),
        default="mir_qbsh",
        help="Eval set to run against (default: mir_qbsh).",
    )
    parser.add_argument(
        "--ranker",
        choices=list(_RANKERS),
        default="dummy",
        help="Ranker to evaluate (default: dummy).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Rank cutoff for Hit@k display (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the dummy ranker (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    ranker = _RANKERS[args.ranker]
    queries = _load_queries(args.dataset)

    if not queries:
        log.warning("No queries loaded — dataset may not be downloaded yet.")
        log.warning("Run `python scripts/v2/setup_eval.py` for instructions.")
        print(
            _format_report(
                ranker_name=args.ranker,
                dataset=args.dataset,
                n_queries=0,
                metrics={},
                elapsed=0.0,
                k=args.k,
            )
        )
        sys.exit(0)

    log.info("Evaluating %d queries with ranker=%r …", len(queries), args.ranker)
    metrics, elapsed = _run_eval(
        queries=queries,
        ranker=ranker,
        k=args.k,
        popularity_scores=None,  # wire in songs.parquet popularity once Phase 1 is done
    )

    print(
        _format_report(
            ranker_name=args.ranker,
            dataset=args.dataset,
            n_queries=len(queries),
            metrics=metrics,
            elapsed=elapsed,
            k=args.k,
        )
    )


if __name__ == "__main__":
    main()
