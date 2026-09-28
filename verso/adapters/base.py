"""Benchmark adapter interface for Verso (Sec. III-E).

Deliberately minimal and benchmark-agnostic: a benchmark only needs to
provide a buggy checkout, a gold checkout, the developer patch between
them, an executable Oracle, and a definition of what counts as
"visible" — which may disclose no tests at all (Sec. III-E's degenerate
case). Nothing here assumes Defects4J's specific tooling; a SWE-bench (or
other) adapter implements the same eight methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import BaseModel


class Task(BaseModel):
    task_id: str
    project: str
    bug_id: str


@dataclass(frozen=True)
class TestRunResult:
    """Sec. II-B: the ground truth a Witness is confirmed against."""

    passed: bool
    detail: str = ""  # compile error / stack trace / assertion message when not passed


@dataclass(frozen=True)
class OracleRunResult:
    passed: bool
    failing_tests: tuple[str, ...] = field(default_factory=tuple)


class ExecutionUnavailableError(RuntimeError):
    """This adapter/environment cannot actually execute or inspect
    something right now — an expected, handled condition (Stage 3
    reports INCONCLUSIVE), not a bug."""


class BenchmarkAdapter(ABC):
    @abstractmethod
    def load_task(self, task_id: str) -> Task:
        raise NotImplementedError

    # --- Visible-only surface: backs the S_vis Extractor's tools (Sec. III-B).
    # Must never expose the developer patch or oracle-only test content.

    @abstractmethod
    def issue_text(self, task: Task) -> str:
        raise NotImplementedError

    @abstractmethod
    def list_visible_files(self, task: Task, subdir: str = "") -> list[str]:
        """Repo-relative paths under `subdir` that a repair system would
        see: the buggy source tree plus whichever tests this benchmark's
        protocol discloses (possibly none)."""
        raise NotImplementedError

    @abstractmethod
    def read_visible_file(self, task: Task, path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def visible_content_contains(self, task: Task, needle: str) -> bool:
        """Whether `needle` (exact text, whitespace-normalized) appears
        anywhere in visible material. Backs the mechanical hidden-ness
        check on a submitted Witness (Sec. II-B): a witness whose
        assertions are lifted from already-visible content can't
        demonstrate the oracle enforcing anything beyond it. A benchmark
        whose visible surface is a handful of curated files can just scan
        them; one whose visible surface is an entire real checkout (e.g.
        SWE-bench, where "visible" is the whole repo at base_commit)
        needs this done as a search over that checkout (e.g. `git grep`)
        rather than by reading every visible file's full content into
        memory/into an LLM prompt — that's the whole reason this is its
        own method instead of always deriving it from
        list_visible_files/read_visible_file."""
        raise NotImplementedError

    # --- Patch surface: backs the S_impl Extractor's tools (Sec. III-B).
    # Must never expose S_vis.

    @abstractmethod
    def developer_patch(self, task: Task) -> str:
        """The unified diff between the buggy and gold revisions."""
        raise NotImplementedError

    @abstractmethod
    def read_buggy_file(self, task: Task, path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def read_gold_file(self, task: Task, path: str) -> str:
        raise NotImplementedError

    # --- Oracle surface: backs the Oracle Search Agent and the Witness
    # Synthesis Agent's mechanical verification (Sec. III-D).

    @abstractmethod
    def run_oracle_on_buggy(self, task: Task) -> OracleRunResult:
        """Run the complete real oracle against the unmodified buggy
        revision. By construction, `passed` should be False (that is
        what makes it "buggy") with a nonempty `failing_tests`."""
        raise NotImplementedError

    @abstractmethod
    def list_oracle_test_files(self, task: Task) -> list[str]:
        """Every test file that is part of the real oracle (visible and
        not), for the Oracle Search Agent to choose among."""
        raise NotImplementedError

    @abstractmethod
    def read_oracle_test_source(self, task: Task, path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def run_candidate_test(self, task: Task, test_source: str, against: str) -> TestRunResult:
        """Compile and run a single candidate test method's source
        against the named revision. `against` is "buggy" or "gold"."""
        raise NotImplementedError
