"""Hermetic in-memory BenchmarkAdapter for dry runs: no filesystem, no
subprocess, no JVM. Every answer is a fixed dict supplied by the caller, so
the pipeline can be exercised end to end without a benchmark environment."""

from __future__ import annotations

from verso.adapters.base import BenchmarkAdapter, OracleRunResult, Task, TestRunResult


class MockAdapter(BenchmarkAdapter):
    def __init__(
        self,
        task: Task,
        *,
        issue_text: str = "",
        visible_files: dict[str, str] | None = None,
        developer_patch: str = "",
        buggy_files: dict[str, str] | None = None,
        gold_files: dict[str, str] | None = None,
        oracle_on_buggy: OracleRunResult | None = None,
        oracle_test_files: dict[str, str] | None = None,
        candidate_test_results: dict[tuple[str, str], TestRunResult] | None = None,
    ) -> None:
        self._task = task
        self._issue_text = issue_text
        self._visible_files = dict(visible_files or {})
        self._developer_patch = developer_patch
        self._buggy_files = dict(buggy_files or {})
        self._gold_files = dict(gold_files or {})
        self._oracle_on_buggy = oracle_on_buggy or OracleRunResult(passed=False, failing_tests=("Trigger::test",))
        self._oracle_test_files = dict(oracle_test_files or {})
        # keyed by (test_source, against) -> TestRunResult; tests set this up per scenario
        self._candidate_test_results = dict(candidate_test_results or {})
        self.calls: list[tuple[str, ...]] = []

    def load_task(self, task_id: str) -> Task:
        if task_id != self._task.task_id:
            raise FileNotFoundError(task_id)
        return self._task

    def issue_text(self, task: Task) -> str:
        self.calls.append(("issue_text",))
        return self._issue_text

    def list_visible_files(self, task: Task, subdir: str = "") -> list[str]:
        self.calls.append(("list_visible_files", subdir))
        return [p for p in self._visible_files if p.startswith(subdir)]

    def read_visible_file(self, task: Task, path: str) -> str:
        self.calls.append(("read_visible_file", path))
        return self._visible_files[path]

    def visible_content_contains(self, task: Task, needle: str) -> bool:
        self.calls.append(("visible_content_contains", needle))
        normalized_needle = " ".join(needle.split())
        corpus = " ".join(self._visible_files.values())
        normalized_corpus = " ".join(corpus.split())
        return normalized_needle in normalized_corpus

    def developer_patch(self, task: Task) -> str:
        self.calls.append(("developer_patch",))
        return self._developer_patch

    def read_buggy_file(self, task: Task, path: str) -> str:
        self.calls.append(("read_buggy_file", path))
        return self._buggy_files[path]

    def read_gold_file(self, task: Task, path: str) -> str:
        self.calls.append(("read_gold_file", path))
        return self._gold_files[path]

    def run_oracle_on_buggy(self, task: Task) -> OracleRunResult:
        self.calls.append(("run_oracle_on_buggy",))
        return self._oracle_on_buggy

    def list_oracle_test_files(self, task: Task) -> list[str]:
        self.calls.append(("list_oracle_test_files",))
        return list(self._oracle_test_files.keys())

    def read_oracle_test_source(self, task: Task, path: str) -> str:
        self.calls.append(("read_oracle_test_source", path))
        return self._oracle_test_files[path]

    def run_candidate_test(self, task: Task, test_source: str, against: str) -> TestRunResult:
        self.calls.append(("run_candidate_test", against))
        key = (test_source, against)
        if key in self._candidate_test_results:
            return self._candidate_test_results[key]
        # Default scripted-by-content convenience when a run doesn't
        # need per-exact-source control: source containing "MATCH" passes
        # on gold and fails on buggy; anything else fails on both.
        if "MATCH" in test_source:
            return TestRunResult(passed=(against == "gold"), detail="" if against == "gold" else "assertion failed")
        return TestRunResult(passed=False, detail="assertion failed")
