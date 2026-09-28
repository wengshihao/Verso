"""Command line interface for Verso.

    verso audit <task_id> --benchmark defects4j --data-dir <dir>
    verso audit-batch --benchmark swebench --instances <file> --tasks <file> --out-dir <dir>

Both commands write the same audit record: the two specifications, the
entailment judgments, the witness with its execution results, and the verdict.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from verso.adapters.base import BenchmarkAdapter
from verso.adapters.bugsinpy import BugsInPyAdapter
from verso.adapters.defects4j import Defects4JAdapter
from verso.adapters.swebench import SweBenchAdapter
from verso.batch import run_batch, summarize
from verso.llm.anthropic_provider import AnthropicProvider
from verso.llm.base import LLMProvider
from verso.pipeline import audit_task
from verso.report import render_json, render_text

BENCHMARKS = ("defects4j", "bugsinpy", "swebench")


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader: sets KEY=VALUE pairs that the shell has not set."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _build_llm() -> LLMProvider:
    return AnthropicProvider()


def _build_adapter(args: argparse.Namespace) -> BenchmarkAdapter:
    if args.benchmark == "defects4j":
        return Defects4JAdapter(args.data_dir)
    if args.benchmark == "bugsinpy":
        return BugsInPyAdapter(args.instances, args.repo_cache)
    if args.benchmark == "swebench":
        return SweBenchAdapter(args.instances)
    raise SystemExit(f"Unknown benchmark: {args.benchmark}")


def _add_benchmark_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    p.add_argument("--data-dir", help="Defects4J fixture directory")
    p.add_argument("--instances", help="instances.jsonl for BugsInPy or SWE-bench")
    p.add_argument("--repo-cache", help="Bare-clone cache directory for BugsInPy")


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="verso")
    sub = parser.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser("audit", help="Audit one task")
    audit_p.add_argument("task_id")
    _add_benchmark_args(audit_p)
    audit_p.add_argument("--json", action="store_true")

    batch_p = sub.add_parser("audit-batch", help="Audit many tasks, one result file each")
    batch_p.add_argument("task_ids", nargs="*")
    _add_benchmark_args(batch_p)
    batch_p.add_argument("--tasks", help="File with one task id per line")
    batch_p.add_argument("--out-dir", required=True)
    batch_p.add_argument("--workers", type=int, default=4)
    batch_p.add_argument("--no-resume", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "audit":
        record = audit_task(args.task_id, llm=_build_llm(), adapter=_build_adapter(args))
        print(render_json(record) if args.json else render_text(record))
        return

    task_ids = list(args.task_ids)
    if args.tasks:
        task_ids += [line.strip() for line in Path(args.tasks).read_text().splitlines() if line.strip()]
    if not task_ids:
        raise SystemExit("Provide task ids or --tasks")

    run_batch(
        task_ids,
        llm_factory=_build_llm,
        adapter_factory=lambda: _build_adapter(args),
        out_dir=args.out_dir,
        max_workers=args.workers,
        resume=not args.no_resume,
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summarize(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
