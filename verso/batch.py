"""Run Verso over many tasks of one benchmark.

Each task is audited in isolation: its own adapter, its own LLM client, its
own result file. A crash in one task never stops the batch, and a rerun
skips tasks that already have a result.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from verso.adapters.base import BenchmarkAdapter
from verso.llm.base import LLMProvider
from verso.llm.counting import CountingProvider
from verso.models import AuditRecord, Verdict
from verso.pipeline import audit_task


@dataclass
class BatchTaskResult:
    task_id: str
    status: str  # "ok" | "error"
    record: AuditRecord | None = None
    error: str | None = None
    duration_seconds: float = 0.0
    llm_calls: int = 0


def _result_path(out_dir: Path, task_id: str) -> Path:
    return out_dir / f"{task_id}.json"


def _write_result(out_dir: Path, result: BatchTaskResult) -> None:
    payload = {
        "task_id": result.task_id,
        "status": result.status,
        "duration_seconds": result.duration_seconds,
        "llm_calls": result.llm_calls,
        "error": result.error,
        "record": result.record.model_dump(mode="json") if result.record else None,
    }
    _result_path(out_dir, result.task_id).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )


def run_batch(
    task_ids: list[str],
    *,
    llm_factory: Callable[[], LLMProvider],
    adapter_factory: Callable[[], BenchmarkAdapter],
    out_dir: str | Path,
    max_workers: int = 4,
    resume: bool = True,
    on_gap_found: Callable[[str], None] | None = None,
) -> list[BatchTaskResult]:
    """Audit every task in `task_ids` and write one JSON record per task.

    `on_gap_found`, if given, is called with the task id right after a
    SPECIFICATION GAP verdict, while that task's execution environment is
    still alive, so a caller can persist whatever the expert review will
    need. An exception there is isolated like an audit error.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for task_id in task_ids:
        if resume and _result_path(out_dir, task_id).exists():
            print(f"[skip] {task_id} already has a result", flush=True)
            continue
        pending.append(task_id)

    def _run_one(task_id: str) -> BatchTaskResult:
        start = time.monotonic()
        llm = CountingProvider(llm_factory())
        adapter = adapter_factory()
        try:
            record = audit_task(task_id, llm=llm, adapter=adapter)
            result = BatchTaskResult(
                task_id=task_id,
                status="ok",
                record=record,
                duration_seconds=time.monotonic() - start,
                llm_calls=llm.total_calls,
            )
            if on_gap_found is not None and record.verdict == Verdict.SPECIFICATION_GAP:
                try:
                    on_gap_found(task_id)
                except Exception as exc:  # noqa: BLE001 - never let this sink a real result
                    print(f"[warn] on_gap_found({task_id}) failed: {exc}", flush=True)
        except Exception as exc:  # noqa: BLE001 - isolate one task's crash from the batch
            result = BatchTaskResult(
                task_id=task_id,
                status="error",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                duration_seconds=time.monotonic() - start,
                llm_calls=llm.total_calls,
            )
        finally:
            # Some adapters hold containers or multi-GB images that outlive a
            # single audit unless torn down. Runs in `finally` so a crash
            # mid-audit still releases them.
            cleanup = getattr(adapter, "cleanup", None)
            if cleanup is not None:
                try:
                    cleanup(adapter.load_task(task_id))
                except Exception:  # noqa: BLE001 - best effort; never mask the real result
                    pass
        _write_result(out_dir, result)
        return result

    results: list[BatchTaskResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, tid): tid for tid in pending}
        for future in as_completed(futures):
            result = future.result()
            status_word = "OK" if result.status == "ok" else "ERROR"
            verdict = result.record.verdict.value if result.record else "-"
            note = result.error.splitlines()[0] if result.error else ""
            print(
                f"[{status_word}] {result.task_id} ({result.duration_seconds:.0f}s, "
                f"{result.llm_calls} LLM calls) verdict={verdict} {note}",
                flush=True,
            )
            results.append(result)

    return results


def summarize(out_dir: str | Path) -> dict:
    """Count verdicts over the result files written by `run_batch`."""
    counts: dict[str, int] = {}
    for path in sorted(Path(out_dir).glob("*.json")):
        payload = json.loads(path.read_text())
        record = payload.get("record")
        key = record["verdict"] if record else f"[{payload.get('status', 'error')}]"
        counts[key] = counts.get(key, 0) + 1
    return counts
