"""Real BenchmarkAdapter for BugsInPy (github.com/soarsmu/BugsInPy),
driving Docker containers for execution and local git archives for
reading. Two deliberate departures from the other two adapters, both
driven by real constraints of the benchmark:

1. Unlike SweBenchAdapter (whole-repo visible surface, one unique
   multi-GB Docker image per task) this adapter's "visible" surface is
   NARROW — the files bug_patch.txt touches plus bug.info's curated
   `test_file` entries — deliberately mirroring Defects4JAdapter's
   curation style (see verso/adapters/defects4j.py:136). BugsInPy was
   added specifically to isolate "language" (Python, like SWE-bench) from
   "curation style" (narrow, like Defects4J) as separate variables in the
   paper's benchmark comparison; see the S_impl surface note below for
   why this ISN'T also true of read_buggy_file/read_gold_file.

2. Unlike both other adapters, execution here needs NO per-task Docker
   image — this environment has no pyenv and can't readily build ancient
   CPython 3.6/3.7 from source on Apple Silicon, but BugsInPy's own
   `bugsinpy-compile` doesn't need a custom image either: it's just
   `python3 -m venv` + `pip install -r requirements.txt` + a project's
   `setup.sh`, on whichever of 7 distinct `python_version`s the bug
   specifies. Official `python:<version>-slim` images cover all 7
   (verified against Docker Hub) and are SHARED across every
   task at that version — 7 total base images, not one per task — so the
   1.37TB-unique-images problem that drove SweBenchAdapter's aggressive
   per-task `cleanup()` doesn't apply here at all; base images are kept
   forever, only per-task containers and local checkouts are torn down.
   `--platform linux/amd64` sidesteps old pinned-version wheels (e.g.
   numpy==1.18.4) having no aarch64 build, same reasoning already used
   for SWE-bench on this machine.

read_buggy_file/read_gold_file (the S_impl Extractor's surface) read
from the FULL real checkout at buggy_commit_id/fixed_commit_id — NOT
restricted to the narrow visible set — exactly mirroring
Defects4JAdapter.read_buggy_file/read_gold_file, which read from
`_checkouts_for` (the real full checkouts), not `_visible_paths` (the
curated fixture subset). Only the VISIBLE surface is narrow; S_impl
always sees the true surrounding implementation.

run_oracle_on_buggy runs against a THIRD checkout/container role,
"oracle" — buggy commit's source with bug.info's test_file(s) overlaid
from the FIXED commit — not the plain "buggy" one. A real container
without this found a bug (youtube-dl__1) whose trigger test PASSED even
against the unmodified buggy checkout: the buggy commit's own test file
simply predates the assertion the fix commit added to actually exercise
the bug. BugsInPy's own `bugsinpy-checkout` script does exactly this
overlay (git-reset to fixed_commit, stash test_file(s) aside, git-reset
to buggy_commit, copy those FIXED test_file(s) back in) for every single
bug — this isn't a per-bug special case, it's how the dataset's
buggy/fixed reproducibility was validated to begin with. See
_local_checkout's "oracle" branch for the mechanics, and
list_oracle_test_files for why the Oracle SEARCH surface (as opposed to
this mechanical ground-truth check) deliberately does NOT get the same
overlay.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

from verso.adapters.base import BenchmarkAdapter, ExecutionUnavailableError, OracleRunResult, Task, TestRunResult

SUBPROCESS_TIMEOUT_SECONDS = 900
WORKDIR_IN_CONTAINER = "/workspace"
BASE_IMAGE_PREFIX = "verso-bugsinpy-base"


def _setuptools_wheel_versions(python_version: str) -> tuple[str, str]:
    """(setuptools, wheel) pins for the base image. 65.5.1/0.38.4 is the
    default (era-appropriate: still vendors its own `packaging`, immune
    to whatever version a task's own requirements.txt pins — see
    _ensure_base_image's docstring) but neither has a distribution for
    Python 3.6 at all (confirmed directly: pip's own candidate list for
    `wheel==0.38.4` on 3.6.9 tops out at 0.37.1) — this dataset's oldest
    python_version (3.6.9, 31 bugs) needs correspondingly older pins."""
    if python_version.startswith("3.6"):
        return "59.6.0", "0.37.1"
    return "65.5.1", "0.38.4"

_CANDIDATE_TEST_TEMPLATE = """\
import pytest

{test_source}
"""

# One shared bare-clone cache per repo (many bugs share a project, e.g.
# 170 pandas bugs), incrementally grown via per-commit shallow fetches.
_GIT_LOCK = threading.Lock()
# Separate from _GIT_LOCK: serializes `docker build` of the 7 shared
# base images (see _ensure_base_image) without also blocking git fetches
# on a slow image build, or vice versa.
_BASE_IMAGE_LOCK = threading.Lock()


def _read_source_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _bare_repo_dir(repo_cache_dir: Path, repo: str) -> Path:
    return repo_cache_dir / (repo.replace("/", "__") + ".git")


def _ensure_bare_clone(repo_cache_dir: Path, repo: str) -> Path:
    bare_dir = _bare_repo_dir(repo_cache_dir, repo)
    if not bare_dir.is_dir():
        repo_cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", "--bare", str(bare_dir)], check=True, capture_output=True)
        subprocess.run(
            ["git", "--git-dir", str(bare_dir), "remote", "add", "origin", f"https://github.com/{repo}.git"],
            check=True,
            capture_output=True,
        )
    return bare_dir


def _archive_commit(bare_dir: Path, commit: str, dest_dir: Path) -> None:
    """Materializes `commit`'s full tree at dest_dir via `git archive`
    (no working `.git`, nothing to later clean up beyond the directory
    itself). Thread-safe: concurrent fetches into the same shared bare
    clone race and fail outright under a concurrent multi-task run."""
    with _GIT_LOCK:
        subprocess.run(
            ["git", "--git-dir", str(bare_dir), "fetch", "--depth", "1", "origin", commit],
            check=True,
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "--git-dir", str(bare_dir), "archive", commit],
        check=True,
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    subprocess.run(["tar", "-x", "-C", str(dest_dir)], input=archive.stdout, check=True, timeout=SUBPROCESS_TIMEOUT_SECONDS)


def _require_docker_cli() -> None:
    try:
        subprocess.run(["docker", "version"], check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise ExecutionUnavailableError(f"`docker` CLI is not available/working: {exc}") from exc


class BugsInPyAdapter(BenchmarkAdapter):
    """`instances_path` holds one JSON object per BugsInPy bug. The adapter
    touches only the real project repos (for checkouts) and Docker Hub (for
    the 7 shared base images)."""

    def __init__(self, instances_path: str, repo_cache_dir: str, platform: str = "linux/amd64") -> None:
        self.instances_path = instances_path
        self.repo_cache_dir = Path(repo_cache_dir)
        self.platform = platform
        self._instances: dict[str, dict] | None = None
        self._task_cache: dict[str, Task] = {}
        # local_checkouts[task_id] = {"buggy": Path, "gold": Path}
        self._local_checkouts: dict[str, dict[str, Path]] = {}
        # containers[task_id] = {"buggy": name, "gold": name}
        self._containers: dict[str, dict[str, str]] = {}
        self._prepared_base_images: set[str] = set()

    # --- instance metadata ---

    def _all_instances(self) -> dict[str, dict]:
        if self._instances is None:
            instances: dict[str, dict] = {}
            with open(self.instances_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        instances[record["task_id"]] = record
            self._instances = instances
        return self._instances

    def _meta(self, task_id: str) -> dict:
        instances = self._all_instances()
        if task_id not in instances:
            raise FileNotFoundError(f"No BugsInPy instance found for task_id={task_id!r}")
        return instances[task_id]

    def load_task(self, task_id: str) -> Task:
        if task_id not in self._task_cache:
            meta = self._meta(task_id)
            self._task_cache[task_id] = Task(task_id=task_id, project=meta["project"], bug_id=meta["bug_id"])
        return self._task_cache[task_id]

    # --- visible-only surface (narrow: patch_files ∪ test_files) ---

    def issue_text(self, task: Task) -> str:
        return self._meta(task.task_id)["issue_text"]

    def _visible_path_set(self, task: Task) -> set[str]:
        # Restricted to paths that actually exist at buggy_commit_id — a
        # real task (spacy__1) found bug.info's own `test_file` naming a
        # file that doesn't exist yet there at all (spacy/tests/
        # test_errors.py was added BY the fix commit, confirmed via
        # `git cat-file -e` against the real repo): nothing genuinely
        # visible to show for a path a repair system checking out the
        # buggy commit would never see in its own tree. Filtering here
        # (once) rather than making every caller of list_visible_files
        # separately handle a KeyError from read_visible_file on a path
        # the list itself just returned.
        meta = self._meta(task.task_id)
        buggy_dir = self._local_checkout(task, "buggy")
        candidates = set(meta["patch_files"]) | set(meta["test_files"])
        return {p for p in candidates if (buggy_dir / p).is_file()}

    def list_visible_files(self, task: Task, subdir: str = "") -> list[str]:
        return [p for p in self._visible_path_set(task) if p.startswith(subdir)]

    def read_visible_file(self, task: Task, path: str) -> str:
        if path not in self._visible_path_set(task):
            raise KeyError(path)
        return self.read_buggy_file(task, path)

    def visible_content_contains(self, task: Task, needle: str) -> bool:
        # Narrow by construction (patch-touched files + curated trigger
        # tests, typically 1-4 files) — a linear scan is cheap, same
        # reasoning as Defects4JAdapter.visible_content_contains.
        normalized_needle = " ".join(needle.split())
        for path in self._visible_path_set(task):
            try:
                content = self.read_visible_file(task, path)
            except KeyError:
                continue
            if normalized_needle in " ".join(content.split()):
                return True
        return False

    # --- patch surface (S_impl: full real checkout, unrestricted) ---

    def developer_patch(self, task: Task) -> str:
        return self._meta(task.task_id)["bug_patch"]

    def _local_checkout(self, task: Task, role: str) -> Path:
        """`role` is "buggy", "gold", or "oracle". "oracle" is buggy
        SOURCE with the FIXED commit's test_files overlaid on top — see
        module docstring's note on run_oracle_on_buggy for why this is
        necessary at all: BugsInPy's own `bugsinpy-checkout` does exactly
        this (git-reset to fixed_commit, stash the test_file(s) aside,
        git-reset to buggy_commit, copy those FIXED test_file(s) back in)
        for every single bug, not a per-bug exception. A real container
        without it found a "bug" whose trigger test PASSED even on the
        buggy checkout — not because the bug doesn't reproduce, but
        because the buggy commit's OWN test file predates the new
        assertion the fix commit added to actually exercise it. Deliberately
        a separate role from "buggy" (never used to back read_buggy_file
        or list_visible_files) so the fixed-commit test content can never
        leak into what looks like ordinary visible/buggy material —
        exactly the same separation SweBenchAdapter keeps between its
        `_buggy_container` and `_oracle_container` (test_patch applied
        only in the latter)."""
        checkouts = self._local_checkouts.setdefault(task.task_id, {})
        if role not in checkouts:
            meta = self._meta(task.task_id)
            if role == "oracle":
                buggy_dir = self._local_checkout(task, "buggy")
                dest = Path(tempfile.mkdtemp(prefix=f"verso-bugsinpy-local-{task.task_id}-oracle-"))
                # symlinks=True: copy symlinks AS links, don't follow
                # them — a real repo (black) has docs/*.md symlinked to
                # a shared source file elsewhere in the tree, and
                # shutil's default (dereference, copy the target's
                # content) raised FileNotFoundError on every one of them
                # for a reason that took a moment to place: `git archive`
                # already materializes symlinks as real symlink files
                # (not resolved content) in buggy_dir, so by the time
                # copytree runs there IS no original target file on disk
                # to dereference — matches what a real checkout looks
                # like anyway (symlinks stay symlinks).
                shutil.copytree(buggy_dir, dest, symlinks=True, dirs_exist_ok=True)
                bare_dir = _ensure_bare_clone(self.repo_cache_dir, meta["repo"])
                for test_path in meta["test_files"]:
                    show = subprocess.run(
                        ["git", "--git-dir", str(bare_dir), "show", f"{meta['fixed_commit_id']}:{test_path}"],
                        capture_output=True,
                        timeout=SUBPROCESS_TIMEOUT_SECONDS,
                    )
                    if show.returncode != 0:
                        continue  # test file didn't exist yet at fixed_commit either — nothing to overlay
                    out_file = dest / test_path
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_bytes(show.stdout)
            else:
                commit = meta["buggy_commit_id"] if role == "buggy" else meta["fixed_commit_id"]
                bare_dir = _ensure_bare_clone(self.repo_cache_dir, meta["repo"])
                dest = Path(tempfile.mkdtemp(prefix=f"verso-bugsinpy-local-{task.task_id}-{role}-"))
                _archive_commit(bare_dir, commit, dest)
            checkouts[role] = dest
        return checkouts[role]

    def _local_cat(self, task: Task, role: str, path: str) -> str:
        full = self._local_checkout(task, role) / path
        if not full.is_file():
            raise KeyError(path)
        return _read_source_text(full.read_bytes())

    def read_buggy_file(self, task: Task, path: str) -> str:
        return self._local_cat(task, "buggy", path)

    def read_gold_file(self, task: Task, path: str) -> str:
        return self._local_cat(task, "gold", path)

    # --- container lifecycle ---

    def _ensure_base_image(self, python_version: str) -> str:
        tag = f"{BASE_IMAGE_PREFIX}:{python_version}"
        if tag in self._prepared_base_images:
            return tag
        _require_docker_cli()
        # `run_batch` gives each task's worker thread its OWN adapter
        # instance (`adapter_factory()` per task), so
        # `self._prepared_base_images` alone can never dedupe a build
        # ACROSS tasks/threads — only `docker image inspect` does, and
        # that check-then-build is a real TOCTOU race: two tasks sharing
        # a python_version (e.g. many youtube-dl bugs, all 3.7.4) that
        # both hit a cold cache at once both start `docker build` for the
        # identical tag, and a real 501-task run saw one of the two
        # simply time out after 900s waiting on Docker's build lock
        # rather than reusing the other's in-progress result. A
        # process-wide lock (not a per-adapter-instance one, and a
        # separate lock from _GIT_LOCK so a slow image build can't also
        # stall unrelated git fetches) actually serializes this.
        with _BASE_IMAGE_LOCK:
            check = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
            if check.returncode != 0:
                self._build_base_image(tag, python_version)
        self._prepared_base_images.add(tag)
        return tag

    def _build_base_image(self, tag: str, python_version: str) -> None:
        dockerfile = (
            f"FROM --platform={self.platform} python:{python_version}-slim\n"
            # Several of the 7 python_versions this dataset needs (e.g.
            # 3.6.9, 3.7.0) sit on a Debian release (buster or older)
            # that's since gone EOL — deb.debian.org 404s on it.
            # archive.debian.org keeps every EOL release forever;
            # Check-Valid-Until=false because archived Release files are
            # (permanently, by design) past their nominal expiry.
            "RUN (sed -i 's|deb.debian.org|archive.debian.org|g; "
            "s|security.debian.org/debian-security|archive.debian.org/debian-security|g' "
            "/etc/apt/sources.list || true) && "
            "apt-get -o Acquire::Check-Valid-Until=false update "
            "-o Acquire::AllowInsecureRepositories=true || true; "
            # Some archived releases (stretch/python:3.7.0,3.7.3) also
            # have expired/missing signing keys in this base image's
            # keyring — real, not a content-integrity concern for a
            # frozen historical archive; --allow-unauthenticated gets
            # past it the same way `apt-get update` failing softly
            # (`|| true` above) does for the sources-list rewrite.
            "apt-get install -y --no-install-recommends --allow-unauthenticated "
            "build-essential git curl pkg-config "
            "&& rm -rf /var/lib/apt/lists/*\n"
            # Pinned, not latest, and for two independent reasons:
            # (1) pytest>=8 pulls in exceptiongroup, whose recent
            # releases use a TypeVar(default=...) broken on these old
            # interpreters (PEP 696 needs a typing_extensions.TypeVar
            # shim exceptiongroup doesn't actually use); pytest 6.2.5
            # predates that dependency entirely. (2) modern setuptools
            # (>=~71) stopped vendoring `packaging` and imports the
            # separately-installed one instead — a real per-task
            # `pip install -r requirements.txt` pinning an old
            # `packaging` (e.g. pandas' `packaging==20.4`) then shadows
            # it, and setuptools' own `_core_metadata.py` calls a
            # `packaging.utils.canonicalize_version(...,
            # strip_trailing_zero=...)` kwarg that old packaging doesn't
            # have — TypeError, unrelated to anything the task itself
            # did wrong. Pinning setuptools to an era-appropriate
            # release (still vendors its own packaging internally) makes
            # it immune to whatever `packaging` version a task's own
            # requirements.txt pins.
            f"RUN pip install -q --upgrade pip && "
            f"pip install -q 'setuptools=={_setuptools_wheel_versions(python_version)[0]}' "
            f"'wheel=={_setuptools_wheel_versions(python_version)[1]}' "
            f"pytest==6.2.5 pytest-json-report==1.5.0\n"
        )
        with tempfile.TemporaryDirectory(prefix="verso-bipy-baseimg-") as ctx:
            (Path(ctx) / "Dockerfile").write_text(dockerfile)
            build = subprocess.run(
                ["docker", "build", "--platform", self.platform, "-t", tag, ctx],
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
            if build.returncode != 0:
                raise ExecutionUnavailableError(f"failed to build base image {tag}: {build.stderr[-2000:]}")

    def _run(self, container: str, cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["docker", "exec", "-w", WORKDIR_IN_CONTAINER, container, "bash", "-lc", cmd],
                check=check,
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionUnavailableError(f"docker exec timed out in {container}: {cmd[:200]}") from exc

    def _start_container(self, task: Task, role: str) -> str:
        meta = self._meta(task.task_id)
        base_image = self._ensure_base_image(meta["python_version"])
        name = f"verso-bipy-{re.sub(r'[^a-zA-Z0-9_.-]', '-', task.task_id)}-{role}"
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--platform",
                self.platform,
                # By default `pip install`/`pip install -e .` builds each
                # package in a FRESH, isolated build-env overlay (PEP
                # 517) that can't see whatever we already installed —
                # real containers hit this twice: pandas' `-e .` pulled
                # its own unpinned (newer, incompatible) Cython into that
                # overlay instead of using our pinned Cython==0.29.19,
                # and spacy's setup.sh re-running `pip install -r
                # requirements.txt` couldn't see the already-installed
                # `cymem`/`preshed` its own Cython (`cimport`) needs at
                # build time. Setting this for the whole container (not
                # just one pip invocation) covers both cases and any
                # ad-hoc `pip install` a project's own setup.sh happens
                # to run.
                "-e",
                "PIP_NO_BUILD_ISOLATION=1",
                "--name",
                name,
                base_image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        local_dir = self._local_checkout(task, role)
        subprocess.run(
            ["docker", "cp", f"{local_dir}/.", f"{name}:{WORKDIR_IN_CONTAINER}"],
            check=True,
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        # Belt-and-suspenders on top of the env var: confirmed directly
        # that PIP_NO_BUILD_ISOLATION alone is NOT reliably honored by
        # every pip install codepath (an identical `pip install -e .`
        # case built in an isolated overlay with only the env var set,
        # and stopped once passed as an explicit `--no-build-isolation`
        # flag — see the per-line requirements install below for the
        # flag). A system pip.conf is the one mechanism that reaches pip
        # invocations we don't construct ourselves (e.g. spacy's own
        # setup.sh re-running `pip install -r requirements.txt`, where
        # there's no argv to add a flag to). Must run AFTER `docker cp`
        # populates WORKDIR_IN_CONTAINER — _run's `-w` fails outright
        # (exit 127) against a working directory that doesn't exist yet.
        self._run(
            name,
            "mkdir -p /etc && printf '[install]\\nno-build-isolation = true\\n' > /etc/pip.conf",
        )

        requirements_text = meta.get("requirements_text", "")
        if requirements_text.strip():
            # One line at a time, not `pip install -r requirements.txt`
            # as a single command: a real container found these captured-
            # from-a-real-dev-machine requirement files sometimes pin a
            # Windows-only package (e.g. scrapy's `pywin32==227`, no
            # Linux wheel or sdist at all) — pip aborts the ENTIRE `-r`
            # batch on that one unresolvable line, silently discarding
            # every other (perfectly installable) pin with it when
            # check=False just swallows the failure. Per-line install
            # means one bad pin only costs that one package.
            #
            # `--no-build-isolation` explicitly, not just the container's
            # PIP_NO_BUILD_ISOLATION env var: a real spacy container hit
            # `preshed/maps.pxd not found` while Cython-compiling one of
            # its OWN requirements.txt entries even with that env var set
            # — pip's env-var-sourced no-build-isolation config isn't
            # reliably honored for every install codepath (confirmed
            # directly: the identical `pip install -e .` case below
            # still built in an isolated overlay with only the env var
            # set, and stopped doing so once passed as an explicit CLI
            # flag). The env var stays set on the container too, as a
            # best-effort net for any ad-hoc `pip install` a project's
            # own setup.sh happens to run where we can't inject a flag.
            lines = [ln.strip() for ln in requirements_text.splitlines()]
            lines = [ln for ln in lines if ln and not ln.startswith("#")]
            # With isolation off, a package needing Cython/numpy at BUILD
            # time now genuinely needs them already installed — no longer
            # true when isolation was on (each package auto-fetched its
            # own declared build-requires into an isolated overlay,
            # regardless of requirements.txt order). A real spacy
            # requirements.txt lists `cymem`/`preshed`/`thinc` (all
            # Cython `cimport`-based) under "Our libraries" and `cython`
            # itself many lines later under "Development dependencies" —
            # cymem's build failed outright (`ModuleNotFoundError: No
            # module named 'Cython'`) before ever reaching that line.
            # Installing common build-time tooling first (wherever it
            # appears in the file), everything else after in its
            # original relative order, fixes this generally rather than
            # special-casing spacy.
            _BUILD_TOOL_RE = re.compile(r"^(cython|numpy|setuptools|wheel|cffi|pybind11)\b", re.IGNORECASE)
            lines.sort(key=lambda ln: 0 if _BUILD_TOOL_RE.match(ln) else 1)
            # An unpinned `cython>=X` (no upper bound) resolves to
            # whatever's newest RIGHT NOW, not whatever was newest when
            # this bug was curated (~2019-2020, going by the
            # python_version range this dataset spans) — Cython 3.0
            # (2023) tightened cdef/cpdef syntax rules significantly, and
            # a real spacy container's OWN .pxd source (not a dependency
            # — spacy's own `vocab.pxd`) failed outright under it
            # ("Variables cannot be declared with 'cpdef'"). Only capped
            # when the line doesn't already pin/bound a version itself —
            # a project that DOES pin exactly (e.g. pandas__1's
            # `Cython==0.29.19`) already gets the historically-correct
            # version and must not be second-guessed.
            _CYTHON_UNPINNED_RE = re.compile(r"^cython\s*(>=|>)", re.IGNORECASE)
            lines = [ln + ",<3" if _CYTHON_UNPINNED_RE.match(ln) and "<" not in ln else ln for ln in lines]
            for line in lines:
                if line.startswith("-e "):
                    # `-e` is pip's own flag — must reach it as a
                    # separate argv token, not folded into one quoted
                    # string with the URL after it.
                    pip_args = f"-e {shlex.quote(line[len('-e ') :].strip())}"
                else:
                    pip_args = shlex.quote(line)
                # --no-deps: requirements.txt is a complete `pip freeze`
                # snapshot — every real dependency (including each
                # package's OWN dependencies) already has its own exact
                # pin somewhere in this same file. Without --no-deps, pip
                # still resolves and silently upgrades an already-pinned
                # package to satisfy whatever a LATER line's own loose
                # runtime `install_requires` declares — confirmed real:
                # pandas__101's `clifford==1.3.1` (far down the file)
                # pulled numpy from the correctly-pinned 1.18.4 to 1.24.4
                # this way, and pandas' OWN C extensions (built afterward
                # against headers 1.24.4 no longer ships in the old
                # layout) failed with an "undeclared identifier" that had
                # nothing to do with the extension file itself.
                # --no-build-isolation only ever governed BUILD-time
                # isolation, never this — a separate, independently
                # necessary flag.
                self._run(name, f"pip install -q --no-build-isolation --no-deps {pip_args}", check=False)

        if meta.get("needs_self_install"):
            # The original requirements.txt pinned an editable install of
            # the bug's own project from a FRESH git clone at some commit
            # (clean_requirements strips that — see its docstring for
            # why); where that line existed at all, its install side
            # effect is sometimes load-bearing (e.g. black's setup.py
            # generates `_black_version.py` via setuptools_scm, which
            # black.py imports at module load time — confirmed by a real
            # container failing with `ModuleNotFoundError: _black_version`
            # without this). `-e .` gets the same effect against the
            # RIGHT commit (the local checkout already in /workspace).
            # That checkout has no `.git` (git-archive'd, not cloned —
            # see _archive_commit), so a project using setuptools_scm for
            # its version (black does) fails outright without real git
            # history to inspect; SETUPTOOLS_SCM_PRETEND_VERSION
            # sidesteps that (unambiguous here — exactly one package is
            # ever being built by this command). `--no-build-isolation`
            # for the same reason as the loop above — a real pandas
            # container's `-e .` pulled a fresh, incompatible Cython into
            # an isolated overlay instead of using our pinned
            # Cython==0.29.19 until this flag was made explicit.
            # `--no-deps` for the same reason as the loop above too — the
            # project's own runtime deps are already satisfied by that
            # loop (requirements.txt is the complete snapshot), so there
            # is nothing correct for pip to add here, only pins it could
            # silently disturb.
            proc = self._run(
                name,
                "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install -q --no-build-isolation --no-deps -e .",
                check=False,
            )
            if proc.returncode != 0:
                raise ExecutionUnavailableError(
                    f"`pip install -e .` failed in {name}: {proc.stderr.decode('utf-8', errors='replace')[-2000:]}"
                )

        setup_cmd = meta.get("setup_cmd", "").strip()
        if setup_cmd:
            # 162 of 501 bugs' setup.sh (nearly every pandas one) builds
            # C extensions with `build_ext -j 0` — distutils' "use every
            # CPU core" — with `max_workers` concurrent tasks each doing
            # this at once, that's `max_workers` full-core compiles
            # fighting over one Docker Desktop VM's memory budget (7.65GB
            # here, well under the 24GB host has). A real 501-task run
            # hit this directly: BugsInPy's alphabetically-sorted task
            # order clusters ~70 pandas bugs together, and around a
            # third of them got bogus compile errors (e.g. a numpy macro
            # reported "undeclared" despite genuinely being in the
            # installed numpy's headers) purely from the contention —
            # confirmed by their timing/clustering, not a real per-bug
            # issue. Capped, not left unbounded — a full `-j 1` serial
            # build would be safest but meaningfully slows every pandas
            # task; `-j 2` keeps some
            # per-task parallelism while bounding aggregate core usage to
            # max_workers * 2 instead of max_workers * all-cores.
            setup_cmd = re.sub(r"-j\s*\d+", "-j 2", setup_cmd)
            proc = self._run(name, setup_cmd, check=False)
            if proc.returncode != 0:
                raise ExecutionUnavailableError(
                    f"setup.sh failed in {name}: {proc.stderr.decode('utf-8', errors='replace')[-2000:]}"
                )
        return name

    def _container_for(self, task: Task, role: str) -> str:
        containers = self._containers.setdefault(task.task_id, {})
        if role not in containers:
            containers[role] = self._start_container(task, role)
        return containers[role]

    # --- oracle surface ---

    def _run_test_lines(self, container: str, run_test_cmd: str) -> tuple[bool, tuple[str, ...]]:
        failing = []
        for line in run_test_cmd.splitlines():
            line = line.strip()
            if not line:
                continue
            proc = self._run(container, line, check=False)
            if proc.returncode != 0:
                failing.append(line)
        return (not failing, tuple(failing))

    def run_oracle_on_buggy(self, task: Task) -> OracleRunResult:
        meta = self._meta(task.task_id)
        # "oracle", not "buggy": see _local_checkout's docstring — the
        # buggy commit's own test file often predates the assertion that
        # actually exercises the fix.
        container = self._container_for(task, "oracle")
        passed, failing = self._run_test_lines(container, meta["run_test_cmd"])
        return OracleRunResult(passed=passed, failing_tests=failing)

    def list_oracle_test_files(self, task: Task) -> list[str]:
        # Deliberately backed by the "buggy" checkout (via
        # read_oracle_test_source -> read_buggy_file below), NOT the
        # "oracle" one — the whole existing test suite is fair game for
        # the Oracle Search Agent to search for a PRE-EXISTING witness,
        # but bug.info's own test_file(s) will read at their pre-fix
        # content here (missing whatever assertion the fix commit itself
        # added — see _local_checkout's "oracle" role docstring). That's
        # intentional, not a gap: content added BY the same fix commit
        # to verify the new behavior isn't a "pre-existing" witness the
        # search should be credited with discovering — same reasoning
        # SweBenchAdapter already applies (its Oracle Search surface
        # doesn't expose test_patch content either). Paths are listed
        # straight off the bare clone (no need for a working tree just
        # to enumerate them).
        meta = self._meta(task.task_id)
        bare_dir = _ensure_bare_clone(self.repo_cache_dir, meta["repo"])
        proc = subprocess.run(
            ["git", "--git-dir", str(bare_dir), "ls-tree", "-r", "--name-only", meta["buggy_commit_id"]],
            check=True,
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
        paths = proc.stdout.decode("utf-8", errors="replace").splitlines()
        return [p for p in paths if "test" in p.lower()]

    def read_oracle_test_source(self, task: Task, path: str) -> str:
        return self.read_buggy_file(task, path)

    def run_candidate_test(self, task: Task, test_source: str, against: str) -> TestRunResult:
        if against not in ("buggy", "gold"):
            return TestRunResult(passed=False, detail=f"invalid `against`: {against!r}")
        container = self._container_for(task, against)

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

        cmd = f"python -m pytest {remote_test_path} -q --json-report --json-report-file={report_path}"
        proc = self._run(container, cmd, check=False)
        report_proc = self._run(container, f"cat -- {report_path}", check=False)
        if report_proc.returncode != 0:
            return TestRunResult(
                passed=False,
                detail=(proc.stdout + proc.stderr).decode("utf-8", errors="replace")[-4000:],
            )
        report = json.loads(report_proc.stdout.decode("utf-8", errors="replace"))
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
        """Per-task containers + local checkouts only — the 7 shared
        base images are cheap (official slim images) and reused by every
        other task at that python_version, so they're deliberately never
        removed here (contrast SweBenchAdapter.cleanup, which MUST
        remove the image every time since there it's 1-per-task)."""
        for name in self._containers.pop(task.task_id, {}).values():
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        for local_dir in self._local_checkouts.pop(task.task_id, {}).values():
            shutil.rmtree(local_dir, ignore_errors=True)
