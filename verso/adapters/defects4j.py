"""Real BenchmarkAdapter for Defects4J, driving the `defects4j` CLI.

Reuses the fixture metadata format for the
visible surface (issue text, curated buggy/visible-test files, the
developer patch) and real checkouts + `defects4j`/`javac`/`java` for
everything execution-related — the oracle, and compiling/running a
candidate test method against either revision.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from verso.adapters.base import BenchmarkAdapter, ExecutionUnavailableError, OracleRunResult, Task, TestRunResult

SUBPROCESS_TIMEOUT_SECONDS = 900


def _read_source_text(path: Path) -> str:
    """Read a file from a real Defects4J checkout. Some old codebases (e.g.
    Commons Compress's TarUtilsTest, which embeds raw high-byte test data
    directly in a string literal) aren't valid UTF-8 even though they
    compiled fine under whatever platform-default encoding javac used at
    the time; fall back to latin-1, which maps every byte 0-255 to a
    codepoint and so never raises, rather than losing/mangling the bytes
    with errors='replace'."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")

_CANDIDATE_TEST_TEMPLATE = """\
import org.junit.Test;
import static org.junit.Assert.*;
{imports}

public class VersoWitnessCandidate {{
{test_source}
}}
"""

# Every agent that writes a candidate is told (and only ever knows how) to
# produce a JUnit 4 `@Test`-annotated method — deliberately, so no subagent
# has to reason about a benchmark's specific toolchain. That is silently
# unusable for at least one project (Commons CLI):
# its `cp.test` classpath carries junit-3.8.1, which has no `org.junit`
# package at all, so `import org.junit.Test;` itself fails to compile for
# EVERY candidate regardless of content — an execution-layer gap that
# looked, from the outside, exactly like "no witness exists" for every
# task in that project. This mechanically translates the same JUnit-4
# method body the agent already wrote into the JUnit 3 shape when needed,
# the same "model never authors the framework boilerplate" principle
# already used for diff hunks (verso/diffutils.py) and the JUnit 4 template
# above.
_JUNIT3_CANDIDATE_TEST_TEMPLATE = """\
{imports}

public class VersoWitnessCandidate extends junit.framework.TestCase {{
    public void testVersoWitnessCandidate() throws Exception {{
{body}
    }}
}}
"""

_JUNIT4_METHOD_RE = re.compile(
    r"@Test\b.*?(?:public\s+)?void\s+\w+\s*\([^)]*\)\s*(?:throws\s+[\w.,\s]+)?\s*\{(.*)\}\s*\Z",
    re.DOTALL,
)


def _is_junit3_only_classpath(classpath: str) -> bool:
    """True if the classpath's junit jar is a 3.x release (no org.junit
    package, hence no @Test/Assert/JUnitCore) — detected from the jar
    filename `defects4j export -p cp.test` returns (e.g.
    `.../junit/jars/junit-3.8.1.jar`), not by trying to load classes."""
    return bool(re.search(r"junit[/-]3\.\d", classpath, re.IGNORECASE))


def _to_junit3_source(test_source: str, imports: str) -> str | None:
    """Extracts the body of the agent's `@Test`-annotated method and
    re-wraps it as a JUnit 3 TestCase. Returns None if `test_source`
    doesn't match the expected single-method shape (caller should then
    report a clear diagnostic rather than silently mis-translating)."""
    match = _JUNIT4_METHOD_RE.search(test_source.strip())
    if match is None:
        return None
    body = match.group(1)
    return _JUNIT3_CANDIDATE_TEST_TEMPLATE.format(imports=imports, body=body)


def _require_defects4j_cli() -> None:
    if shutil.which("defects4j") is None:
        raise ExecutionUnavailableError(
            "The `defects4j` CLI was not found on PATH. Install Java 11 + Defects4J "
            "(see docs/defects4j_setup.md) or use MockAdapter for hermetic tests."
        )


class Defects4JAdapter(BenchmarkAdapter):
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._meta_cache: dict[str, dict] = {}
        # task_id -> (buggy_checkout_dir, gold_checkout_dir); created lazily,
        # reused across every tool call for that task within this process.
        self._checkouts: dict[str, tuple[Path, Path]] = {}
        self._oracle_cache: dict[str, OracleRunResult] = {}
        self._compiled: set[Path] = set()

    # --- fixture metadata (visible surface + developer patch) ---

    def _task_dir(self, task_id: str) -> Path:
        task_dir = self.data_dir / task_id
        if not task_dir.is_dir():
            raise FileNotFoundError(f"No fixture found for task_id={task_id!r} under {self.data_dir}")
        return task_dir

    def _meta(self, task_id: str) -> dict:
        if task_id not in self._meta_cache:
            self._meta_cache[task_id] = json.loads((self._task_dir(task_id) / "meta.json").read_text())
        return self._meta_cache[task_id]

    def load_task(self, task_id: str) -> Task:
        meta = self._meta(task_id)
        return Task(task_id=meta["task_id"], project=meta["project"], bug_id=meta["bug_id"])

    def issue_text(self, task: Task) -> str:
        meta = self._meta(task.task_id)
        return (self._task_dir(task.task_id) / meta["issue_text_file"]).read_text()

    def _visible_paths(self, task: Task) -> dict[str, str]:
        meta = self._meta(task.task_id)
        task_dir = self._task_dir(task.task_id)
        paths: dict[str, str] = {}
        for repo_path, fixture_path in {**meta["buggy_files"], **meta.get("visible_tests", {})}.items():
            paths[repo_path] = fixture_path
        return {repo_path: str(task_dir / fixture_path) for repo_path, fixture_path in paths.items()}

    def list_visible_files(self, task: Task, subdir: str = "") -> list[str]:
        return [p for p in self._visible_paths(task) if p.startswith(subdir)]

    def read_visible_file(self, task: Task, path: str) -> str:
        paths = self._visible_paths(task)
        if path not in paths:
            raise KeyError(path)
        return _read_source_text(Path(paths[path]))

    def visible_content_contains(self, task: Task, needle: str) -> bool:
        # This adapter's visible surface is always a handful of curated
        # files (the modified classes + trigger tests), so reading them
        # all is cheap — no need for a smarter search than this.
        normalized_needle = " ".join(needle.split())
        for path in self._visible_paths(task):
            content = self.read_visible_file(task, path)
            if normalized_needle in " ".join(content.split()):
                return True
        return False

    def developer_patch(self, task: Task) -> str:
        meta = self._meta(task.task_id)
        return (self._task_dir(task.task_id) / meta["developer_patch_file"]).read_text()

    # --- real checkouts ---

    def _checkout_revision(self, task: Task, workdir: Path, variant: str) -> None:
        _require_defects4j_cli()
        try:
            subprocess.run(
                ["defects4j", "checkout", "-p", task.project, "-v", f"{task.bug_id}{variant}", "-w", str(workdir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ExecutionUnavailableError(f"`defects4j checkout` failed for {task.task_id}/{variant}: {exc}") from exc

    def _checkouts_for(self, task: Task) -> tuple[Path, Path]:
        if task.task_id not in self._checkouts:
            root = Path(tempfile.mkdtemp(prefix=f"verso-{task.task_id}-"))
            buggy_dir, gold_dir = root / "buggy", root / "gold"
            self._checkout_revision(task, buggy_dir, "b")
            self._checkout_revision(task, gold_dir, "f")
            self._checkouts[task.task_id] = (buggy_dir, gold_dir)
        return self._checkouts[task.task_id]

    def read_buggy_file(self, task: Task, path: str) -> str:
        buggy_dir, _ = self._checkouts_for(task)
        full = buggy_dir / path
        if not full.is_file():
            raise KeyError(path)
        return _read_source_text(full)

    def read_gold_file(self, task: Task, path: str) -> str:
        _, gold_dir = self._checkouts_for(task)
        full = gold_dir / path
        if not full.is_file():
            raise KeyError(path)
        return _read_source_text(full)

    def _defects4j_export(self, workdir: Path, prop: str) -> str:
        try:
            return subprocess.run(
                ["defects4j", "export", "-p", prop, "-w", str(workdir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            ).stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ExecutionUnavailableError(f"`defects4j export -p {prop}` failed: {exc}") from exc

    def _compile(self, workdir: Path) -> None:
        subprocess.run(
            ["defects4j", "compile"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )

    def _ensure_compiled(self, workdir: Path) -> None:
        """Memoized `_compile`. Without it, `run_candidate_test` can
        export `cp.test` (and compile a candidate against it) for a
        checkout that has never been through `defects4j compile`
        — for at least one project (Commons CLI) that leaves JUnit itself
        off the resulting classpath (its build only resolves/stages test
        dependencies as a side effect of compiling), making every witness
        candidate fail with "package org.junit does not exist" regardless
        of content. `run_oracle_on_buggy` already compiled `buggy_dir` for
        its own purpose, but `gold_dir` was never compiled by anything —
        this makes every checkout get compiled at least once before its
        classpath is trusted for anything, cached so repeated `run_test`
        calls against the same checkout don't recompile every time."""
        if workdir in self._compiled:
            return
        self._compile(workdir)
        self._compiled.add(workdir)

    # --- oracle surface ---

    def run_oracle_on_buggy(self, task: Task) -> OracleRunResult:
        if task.task_id in self._oracle_cache:
            return self._oracle_cache[task.task_id]
        buggy_dir, _ = self._checkouts_for(task)
        self._ensure_compiled(buggy_dir)
        try:
            subprocess.run(
                ["defects4j", "test", "-r"],
                cwd=buggy_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.CalledProcessError as exc:
            raise ExecutionUnavailableError(f"`defects4j test -r` failed to run: {exc}") from exc
        failing_tests_file = buggy_dir / "failing_tests"
        failing: list[str] = []
        if failing_tests_file.exists():
            for line in failing_tests_file.read_text().splitlines():
                if line.startswith("--- "):
                    failing.append(line[len("--- ") :].strip())
        result = OracleRunResult(passed=not failing, failing_tests=tuple(failing))
        self._oracle_cache[task.task_id] = result
        return result

    def list_oracle_test_files(self, task: Task) -> list[str]:
        buggy_dir, _ = self._checkouts_for(task)
        test_src_dir = self._defects4j_export(buggy_dir, "dir.src.tests")
        relevant = self._defects4j_export(buggy_dir, "tests.relevant")
        classes = sorted({line.split("::")[0] for line in relevant.splitlines() if line.strip()})
        paths = []
        for cls in classes:
            rel_path = f"{test_src_dir}/{cls.replace('.', '/')}.java"
            if (buggy_dir / rel_path).is_file():
                paths.append(rel_path)
        return paths

    def read_oracle_test_source(self, task: Task, path: str) -> str:
        buggy_dir, _ = self._checkouts_for(task)
        full = buggy_dir / path
        if not full.is_file():
            raise KeyError(path)
        return _read_source_text(full)

    def run_candidate_test(self, task: Task, test_source: str, against: str) -> TestRunResult:
        if against not in ("buggy", "gold"):
            return TestRunResult(passed=False, detail=f"invalid `against`: {against!r}")
        buggy_dir, gold_dir = self._checkouts_for(task)
        workdir = buggy_dir if against == "buggy" else gold_dir
        self._ensure_compiled(workdir)

        imports = "\n".join(sorted(set(re.findall(r"\bimport\s+[\w.]+;", test_source))))
        cleaned_source = re.sub(r"^\s*import\s+[\w.]+;\s*$", "", test_source, flags=re.MULTILINE)

        # cp.compile only covers main classes (no JUnit) — that's fine for
        # a JUnit-based candidate test
        # needs cp.test, which brings in the actual JUnit jar (and already
        # includes the compiled main/test classes dirs itself). Requires
        # workdir to have actually been compiled at least once already
        # (see _ensure_compiled above) — for some projects cp.test's JUnit
        # entry doesn't exist on disk until a real `defects4j compile` has
        # run.
        classpath = self._defects4j_export(workdir, "cp.test")
        missing_jars = [p for p in classpath.split(":") if p.endswith(".jar") and not Path(p).is_file()]
        if missing_jars:
            # A real diagnosis found this happen for a legacy project
            # (Commons CLI, junit-3.8.1): `defects4j compile` creates the
            # target/lib/<dep>/jars/ directory entry in `cp.test` but the
            # actual jar never gets fetched into it — reachable over the
            # network in general (Maven Central resolves fine), but this
            # specific project's Ant/Ivy config apparently points at a
            # dead legacy repository for this one dependency. That's an
            # environment/toolchain gap, not something a witness — of any
            # content — could ever work around, so it must surface as
            # "execution unavailable", not a confusing raw javac error.
            raise ExecutionUnavailableError(
                f"cp.test names {len(missing_jars)} jar(s) that don't actually exist on disk "
                f"(e.g. {missing_jars[0]}) — likely an unreachable/misconfigured dependency in this "
                f"project's build, not something any candidate test content could work around."
            )
        junit3 = _is_junit3_only_classpath(classpath)
        if junit3:
            source = _to_junit3_source(cleaned_source, imports)
            if source is None:
                return TestRunResult(
                    passed=False,
                    detail="this project's oracle only has JUnit 3 (no org.junit package), and the candidate's "
                    "@Test method could not be mechanically translated to a JUnit 3 TestCase — check it is a "
                    "single, complete @Test-annotated method.",
                )
            runner = ["junit.textui.TestRunner", "VersoWitnessCandidate"]
        else:
            source = _CANDIDATE_TEST_TEMPLATE.format(imports=imports, test_source=cleaned_source)
            runner = ["org.junit.runner.JUnitCore", "VersoWitnessCandidate"]

        with tempfile.TemporaryDirectory(prefix=f"verso-witness-{task.task_id}-") as tmp:
            probe_dir = Path(tmp)
            probe_file = probe_dir / "VersoWitnessCandidate.java"
            probe_file.write_text(source)

            try:
                subprocess.run(
                    ["javac", "-cp", classpath, "-d", str(probe_dir), str(probe_file)],
                    cwd=probe_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=SUBPROCESS_TIMEOUT_SECONDS,
                )
            except subprocess.CalledProcessError as exc:
                return TestRunResult(passed=False, detail=f"compile failed: {exc.stderr or exc.stdout}")
            except subprocess.TimeoutExpired as exc:
                return TestRunResult(passed=False, detail=f"compile timed out: {exc}")

            try:
                run = subprocess.run(
                    ["java", "-cp", f"{probe_dir}:{classpath}", *runner],
                    cwd=probe_dir,
                    capture_output=True,
                    text=True,
                    timeout=SUBPROCESS_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                return TestRunResult(passed=False, detail=f"run timed out: {exc}")

            if run.returncode == 0:
                return TestRunResult(passed=True)
            return TestRunResult(passed=False, detail=(run.stdout + run.stderr)[-4000:])
