# Verso

Replication package for *Requirements Nobody Stated: Auditing Specification Gaps in Program Repair Benchmarks*.

A **specification gap** is a requirement that a benchmark's oracle enforces but that
nothing visible to a repair system states. **Verso** is an agentic auditor that finds
such gaps and, for each one, produces an executable **witness**: a test that passes on
the developer's patch, fails on the buggy program, and checks behavior that appears
nowhere in the visible information.

This repository contains Verso, the audit records of all 2,355 audited tasks, the
expert labels, and the scripts that produce every table and figure in the paper.

Two kinds of data are kept strictly apart. Everything under `audits/` is Verso's own
output, produced by its agents. Everything under `expert_labels/` was produced by the
six human experts: they read each case and wrote the Yes/No answers themselves. No
label in `expert_labels/` comes from a model.

## Layout

```
verso/            the auditor (Sec. III of the paper)
audits/           one JSON record per audited task, 2,355 in total
expert_labels/    the expert validation records (Sec. IV-D)
data/swebench/    the task manifests and external data used in the analysis
scripts/          the analysis scripts that produce the tables and figures
tables/           output of scripts/make_tables.py
figures/          output of the five figure scripts
```

### `audits/`

One file per task, grouped by benchmark: `defects4j/` (854), `bugsinpy/` (501),
`swebench_unverified/` (500), `swebench_verified/` (500). Each file holds the full
audit record:

| field | meaning |
| --- | --- |
| `s_vis`, `s_impl` | the two independently extracted specifications |
| `entailments` | the judge's verdict for each requirement of `s_impl` |
| `target_requirement_id` | the unentailed requirement the witness search targeted |
| `witness` | the witness, its source, and its results on the gold and buggy programs |
| `notebook` | the shared task notebook, in the order the agents wrote it |
| `verdict` | `SPECIFICATION GAP`, `NO EVIDENCE OF GAP`, or `INCONCLUSIVE` |

A `SPECIFICATION GAP` verdict is Verso's report, not the final result. The 1,278
reported gaps are exactly the cases the experts reviewed.

### `expert_labels/`

Six experts reviewed all 1,278 reported gaps and answered the six criteria C1–C6 of
Table IV with Yes or No. Two experts labeled each case independently, and a third
resolved every disagreement.

| file | contents |
| --- | --- |
| `assignment.csv` | the two labelers and the adjudicator assigned to each case |
| `expert_E1.csv` … `expert_E6.csv` | each expert's own independent labels |
| `adjudication.csv` | the 573 disagreements and the third expert's decision |
| `final_labels.csv` | the resolved C1–C6 labels and the confirmed flag for all 1,278 cases |
| `gap_types.csv` | the behavior type and edge-case flag of each of the 720 confirmed gaps |

Every row's `audit_record` column points to the audit record the expert reviewed.

### `data/swebench/`

`swebench_verified_500_manifest.json` and `swebench_unverified_500_manifest.json` list
the audited tasks of each SWE-bench split, and `swebench_unverified_1199_pool.json`
lists the pool the Unverified sample was drawn from.
`leaderboard_results_15systems.json` holds the per-task outcomes the 15 systems
submitted to the SWE-bench Verified leaderboard (RQ4),
`openai_annotations.csv` the published annotations from the screening of SWE-bench
Verified (Fig. 8), and `issue_lengths.csv` the issue length of each SWE-bench task
(Fig. 7).

## Reproducing the results

```bash
pip install -e ".[analysis]"
python scripts/make_tables.py            # Tables V and VII, and the counts behind Fig. 4
python scripts/verify_expert_labels.py   # Cohen's kappa of Table IV; checks final_labels.csv
python scripts/fig4_gap_types.py         # Fig. 4
python scripts/fig5_systems.py           # Fig. 5
python scripts/fig6_unresolved_tasks.py  # Fig. 6
python scripts/fig7_gap_origins.py       # Fig. 7
python scripts/fig8_annotator_ratings.py # Fig. 8
```

Every script reads only the files in this repository and needs no API key. The numbers
reported in Sec. V are recomputed from the audit records and the expert labels each
time, not read from a stored summary. `verify_expert_labels.py` rebuilds each case
from the two independent sheets plus the adjudicator's decision and checks the result
against `final_labels.csv`.

## Running Verso

Verso audits one task at a time and needs the benchmark's own environment, because
every candidate witness is actually executed: the `defects4j` CLI and a JDK for
Defects4J, and Docker for BugsInPy and SWE-bench. It also needs a model endpoint that
speaks the Anthropic Messages protocol. Copy `.env.example` to `.env` and fill it in;
the paper's runs used DeepSeek-V4-Flash.

```bash
pip install -e .
verso audit astropy__astropy-13398 --benchmark swebench --instances instances.jsonl
verso audit-batch --benchmark swebench --instances instances.jsonl \
      --tasks tasks.txt --out-dir audits/rerun
```

`--instances` is a JSONL export of the benchmark's own instance records, one object
per line. Defects4J instead takes `--data-dir`, a directory with one folder per task
holding that task's issue text, visible files, and developer patch. `audit-batch`
writes one record per task in the format of `audits/`, isolates failures, and resumes
a run that stopped.

Runs are not bit-for-bit reproducible: the agents call a language model, and benchmark
environments change over time. `audits/` holds the records the paper reports.
