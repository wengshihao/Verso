"""Real BenchmarkAdapter for SWE-bench, driving official per-instance
Docker images (`swebench` package convention: `swebench/sweb.eval.<arch>.
<instance_id>:latest`) via the `docker` CLI.

Design departs from Defects4JAdapter in one deliberate way: "visible"
here is the ENTIRE repository at base_commit, not a hand-curated handful
of files. That's not a simplification for Verso's convenience — it's what
SWE-bench's own evaluation protocol actually discloses to an agent (full
repo access, explore freely); `test_patch` (the FAIL_TO_PASS/PASS_TO_PASS
additions) is the one thing genuinely withheld until grading, so that's
the only hidden-witness surface. See Sec. II-B of the paper.

Because "visible" can be thousands of files, `visible_content_contains`
is answered with `git grep` rather than by concatenating every visible
file into one string (see the docstring on
BenchmarkAdapter.visible_content_contains for why that distinction
exists at all).

All *reading* of visible material (list/read/grep) runs against a LOCAL
`docker cp` checkout, not `docker exec` into the container — a profiled
real audit found 1,699 docker exec calls eating 65% of total wall time
(the container itself was only ever needed for the couple of exec calls
that actually run pytest; everything else was S_vis Extractor / Oracle
Search Agent exploring/grepping the checkout, which a local copy serves
just as well and far faster, since `docker exec` pays a real subprocess
fork/exec cost, worse still under x86_64-on-arm64 emulation,
on every single call, and there'd be well over a thousand of those on
a repo with ~2,000 tracked files). Only genuine code EXECUTION
(`run_oracle_on_buggy`, `run_candidate_test`) still goes through the
container — there's no way around actually running pytest inside it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from verso.adapters.base import BenchmarkAdapter, ExecutionUnavailableError, OracleRunResult, Task, TestRunResult

SUBPROCESS_TIMEOUT_SECONDS = 900
REPO_ROOT_IN_CONTAINER = "/testbed"

_CANDIDATE_TEST_TEMPLATE = """\
import pytest

{test_source}
"""


def _read_source_text(raw: bytes) -> str:
    """Same UTF-8-first/latin-1-fallback policy as the Defects4J adapter
    and fixture builder — a real checkout can contain source files that
    aren't valid UTF-8 (see verso/adapters/defects4j.py's docstring for
    the concrete case that motivated this)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _require_docker_cli() -> None:
    try:
        subprocess.run(["docker", "version"], check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise ExecutionUnavailableError(f"`docker` CLI is not available/working: {exc}") from exc


class SweBenchAdapter(BenchmarkAdapter):
    """`instances_path` is a local JSONL cache of SWE-bench instance
    records exported from the dataset. The adapter never hits the Hugging
    Face Hub itself, so a batch run isn't coupled to network
    availability/rate limits beyond that export and `docker pull`."""

    def __init__(self, instances_path: str, platform: str = "linux/amd64") -> None:
        self.instances_path = instances_path
        self.platform = platform
        self._instances: dict[str, dict] | None = None
        self._task_cache: dict[str, Task] = {}
        # containers[task_id] = {"buggy": name, "oracle": name, "gold": name}
        self._containers: dict[str, dict[str, str]] = {}
        # local_checkouts[task_id] = {"buggy": Path, "gold": Path} — see
        # module docstring for why reads go through these, not docker exec.
        self._local_checkouts: dict[str, dict[str, Path]] = {}
        self._pulled_images: set[str] = set()

    # --- instance metadata ---

    def _all_instances(self) -> dict[str, dict]:
        if self._instances is None:
            instances: dict[str, dict] = {}
            with open(self.instances_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    instances[record["instance_id"]] = record
            self._instances = instances
        return self._instances

    def _meta(self, task_id: str) -> dict:
        instances = self._all_instances()
        if task_id not in instances:
            raise FileNotFoundError(f"No SWE-bench instance found for task_id={task_id!r}")
        return instances[task_id]

    def load_task(self, task_id: str) -> Task:
        if task_id not in self._task_cache:
            meta = self._meta(task_id)
            self._task_cache[task_id] = Task(task_id=task_id, project=meta["repo"], bug_id=meta["instance_id"])
        return self._task_cache[task_id]

    # --- visible-only surface ---

    def issue_text(self, task: Task) -> str:
        return self._meta(task.task_id)["problem_statement"]

    def _run(self, container: str, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["docker", "exec", container, "bash", "-lc", cmd],
                check=check,
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionUnavailableError(f"docker exec timed out in {container}: {cmd[:200]}") from exc

    def _local_checkout(self, task: Task, role: str) -> Path:
        """The `role` container's /testbed, `docker cp`'d out once and
        cached — see module docstring. `role` is "buggy" or "gold"."""
        checkouts = self._local_checkouts.setdefault(task.task_id, {})
        if role not in checkouts:
            container = self._buggy_container(task) if role == "buggy" else self._gold_container(task)
            local_dir = Path(tempfile.mkdtemp(prefix=f"verso-swebench-local-{task.task_id}-{role}-"))
            # The trailing "/." on the source copies CONTENTS of
            # /testbed into local_dir, not /testbed itself as a
            # subdirectory of it.
            subprocess.run(
                ["docker", "cp", f"{container}:{REPO_ROOT_IN_CONTAINER}/.", str(local_dir)],
                check=True,
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
            checkouts[role] = local_dir
        return checkouts[role]

    def _local_cat(self, task: Task, role: str, path: str) -> str:
        full = self._local_checkout(task, role) / path
        if not full.is_file():
            raise KeyError(path)
        return _read_source_text(full.read_bytes())

    def list_visible_files(self, task: Task, subdir: str = "") -> list[str]:
        local_dir = self._local_checkout(task, "buggy")
        proc = subprocess.run(
            ["git", "-C", str(local_dir), "ls-files"],
            check=True,
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        files = [line for line in proc.stdout.decode("utf-8", errors="replace").splitlines() if line]
        return [p for p in files if p.startswith(subdir)]

    def read_visible_file(self, task: Task, path: str) -> str:
        return self._local_cat(task, "buggy", path)

    def visible_content_contains(self, task: Task, needle: str) -> bool:
        local_dir = self._local_checkout(task, "buggy")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(needle)
            needle_path = f.name
        try:
            # -F: fixed string, not regex; -f: read the pattern from a
            # file (avoids shell-quoting issues with the needle's own
            # content); -q: exit-status only.
            proc = subprocess.run(
                ["git", "-C", str(local_dir), "grep", "-qF", "-f", needle_path],
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
            return proc.returncode == 0
        finally:
            Path(needle_path).unlink(missing_ok=True)

    # --- patch surface ---

    def developer_patch(self, task: Task) -> str:
        return self._meta(task.task_id)["patch"]

    def read_buggy_file(self, task: Task, path: str) -> str:
        return self._local_cat(task, "buggy", path)

    def read_gold_file(self, task: Task, path: str) -> str:
        return self._local_cat(task, "gold", path)

    # --- container lifecycle ---

    def _ensure_image(self, image: str) -> None:
        if image in self._pulled_images:
            return
        _require_docker_cli()
        check = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
        if check.returncode != 0:
            pull = subprocess.run(
                ["docker", "pull", "--platform", self.platform, image],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
            if pull.returncode != 0:
                raise ExecutionUnavailableError(f"`docker pull {image}` failed: {pull.stderr}")
        self._pulled_images.add(image)

    def _start_container(self, task: Task, role: str) -> str:
        meta = self._meta(task.task_id)
        image = meta["image"]
        self._ensure_image(image)
        name = f"verso-swebench-{re.sub(r'[^a-zA-Z0-9_.-]', '-', task.task_id)}-{role}"
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(
            ["docker", "run", "-d", "--platform", self.platform, "--name", name, image, "sleep", "infinity"],
            check=True,
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        # Official images don't ship pytest-json-report; a plain-text
        # pass/fail parse would have to be re-derived per repo (this
        # dataset's own `log_parser` field is a different parser per
        # project for exactly that reason) — one pip install buys a
        # single, reliable machine-readable format for every repo instead.
        self._run(name, "source /opt/miniconda3/bin/activate testbed && pip install -q pytest-json-report")
        return name

    def _apply_patch(self, container: str, patch_text: str) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch_text)
            local_path = f.name
        try:
            remote_path = f"/tmp/verso-{uuid.uuid4().hex}.patch"
            subprocess.run(["docker", "cp", local_path, f"{container}:{remote_path}"], check=True, capture_output=True)
            proc = self._run(container, f"cd {REPO_ROOT_IN_CONTAINER} && git apply -v {remote_path}", check=False)
            if proc.returncode != 0:
                raise ExecutionUnavailableError(
                    f"`git apply` failed in {container}: {proc.stderr.decode('utf-8', errors='replace')}"
                )
        finally:
            Path(local_path).unlink(missing_ok=True)

    def _buggy_container(self, task: Task) -> str:
        containers = self._containers.setdefault(task.task_id, {})
        if "buggy" not in containers:
            containers["buggy"] = self._start_container(task, "buggy")
        return containers["buggy"]

    def _gold_container(self, task: Task) -> str:
        containers = self._containers.setdefault(task.task_id, {})
        if "gold" not in containers:
            name = self._start_container(task, "gold")
            self._apply_patch(name, self._meta(task.task_id)["patch"])
            containers["gold"] = name
        return containers["gold"]

    def _oracle_container(self, task: Task) -> str:
        """Buggy source + the test_patch's new/changed test content
        applied, so the real FAIL_TO_PASS/PASS_TO_PASS tests are actually
        runnable — but kept as a THIRD, separate container from
        `_buggy_container`, never used to back read_visible_file/
        read_buggy_file, so applying test_patch here can never leak
        hidden test content into what looks like visible material."""
        containers = self._containers.setdefault(task.task_id, {})
        if "oracle" not in containers:
            name = self._start_container(task, "oracle")
            self._apply_patch(name, self._meta(task.task_id)["test_patch"])
            containers["oracle"] = name
        return containers["oracle"]

    # --- oracle surface ---

    def run_oracle_on_buggy(self, task: Task) -> OracleRunResult:
        meta = self._meta(task.task_id)
        container = self._oracle_container(task)
        test_ids = list(meta["FAIL_TO_PASS"]) + list(meta["PASS_TO_PASS"])
        report_path = f"/tmp/verso-report-{uuid.uuid4().hex}.json"
        quoted_ids = " ".join(json.dumps(t) for t in test_ids)
        cmd = (
            f"source /opt/miniconda3/bin/activate testbed && cd {REPO_ROOT_IN_CONTAINER} && "
            f"python -m pytest {quoted_ids} -q --json-report --json-report-file={report_path}"
        )
        self._run(container, cmd, check=False)
        report = self._read_json_report(container, report_path)
        failing = tuple(
            t["nodeid"] for t in report.get("tests", []) if t.get("outcome") not in ("passed", "xfailed")
        )
        return OracleRunResult(passed=not failing, failing_tests=failing)

    def _read_json_report(self, container: str, report_path: str) -> dict:
        proc = self._run(container, f"cat -- {report_path}", check=False)
        if proc.returncode != 0:
            raise ExecutionUnavailableError(
                f"pytest --json-report did not produce a report at {report_path} in {container} "
                "(pytest-json-report plugin missing from this image?)"
            )
        return json.loads(proc.stdout.decode("utf-8", errors="replace"))

    def list_oracle_test_files(self, task: Task) -> list[str]:
        # SWE-bench discloses the whole repo, so every existing test file
        # is fair game for the Oracle Search Agent to check — restricted
        # to test-shaped paths purely so the agent isn't handed thousands
        # of unrelated source files under this tool.
        return [p for p in self.list_visible_files(task) if "test" in p.lower()]

    def read_oracle_test_source(self, task: Task, path: str) -> str:
        return self.read_visible_file(task, path)

    def run_candidate_test(self, task: Task, test_source: str, against: str) -> TestRunResult:
        if against not in ("buggy", "gold"):
            return TestRunResult(passed=False, detail=f"invalid `against`: {against!r}")
        container = self._buggy_container(task) if against == "buggy" else self._gold_container(task)

        source = _CANDIDATE_TEST_TEMPLATE.format(test_source=test_source)
        remote_test_path = f"/tmp/verso_witness_{uuid.uuid4().hex}.py"
        report_path = f"/tmp/verso-report-{uuid.uuid4().hex}.json"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            local_path = f.name
        try:
            subprocess.run(
                ["docker", "cp", local_path, f"{container}:{remote_test_path}"], check=True, capture_output=True
            )
        finally:
            Path(local_path).unlink(missing_ok=True)

        cmd = (
            f"source /opt/miniconda3/bin/activate testbed && cd {REPO_ROOT_IN_CONTAINER} && "
            f"python -m pytest {remote_test_path} -q --json-report --json-report-file={report_path}"
        )
        proc = self._run(container, cmd, check=False)
        try:
            report = self._read_json_report(container, report_path)
        except ExecutionUnavailableError:
            # No report at all usually means a collection error (e.g. a
            # SyntaxError/ImportError in the candidate) rather than a
            # normal assertion failure — surface pytest's own stderr/
            # stdout as the detail rather than raising, since this is a
            # legitimate "candidate FAILED" outcome, not an environment
            # problem.
            return TestRunResult(
                passed=False,
                detail=(proc.stdout + proc.stderr).decode("utf-8", errors="replace")[-4000:],
            )
        tests = report.get("tests", [])
        if not tests:
            return TestRunResult(
                passed=False,
                detail="pytest collected no tests from the candidate (check it defines at least one test_* function).",
            )
        passed = all(t.get("outcome") in ("passed", "xfailed") for t in tests)
        detail = "" if passed else json.dumps([t for t in tests if t.get("outcome") not in ("passed", "xfailed")])[:4000]
        return TestRunResult(passed=passed, detail=detail)

    # --- cleanup ---

    def cleanup(self, task: Task) -> None:
        """Not part of BenchmarkAdapter — SWE-bench's containers AND
        images are real, disk-heavy resources (unlike Defects4J's
        tempfile checkouts, which the OS reclaims), so a batch runner
        MUST call this once a task's audit is done, not just at the end
        of a whole run: a survey of all 1,000 Verified+Unverified
        instances found 1,370 GB of (compressed) unique images — every
        instance gets its own image, none are shared — which would need
        an estimated ~5 TB once pulled and extracted if left cached, far
        more than this machine's ~440 GB free. Removing the image after
        each task keeps steady-state disk usage bounded by concurrency
        (a handful of images in flight at once), not by how many tasks
        have been audited so far.

        Safe to remove the image unconditionally: the 1-image-per-
        instance property above means no other task_id could be sharing
        it (confirmed against the full 1,000-instance manifest, not
        assumed)."""
        for name in self._containers.pop(task.task_id, {}).values():
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        image = self._meta(task.task_id)["image"]
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True)
        self._pulled_images.discard(image)
        for local_dir in self._local_checkouts.pop(task.task_id, {}).values():
            shutil.rmtree(local_dir, ignore_errors=True)
