# AWS Production Inference Deployment Audit Report

**Date**: September 26, 2026  
**Target Module**: AWS Production Inference Pipeline (`code/business_entity_resolution/src/run_full_inference.py` & `run_aws_inference.sh`)  
**Scope**: AWS Linux/EC2 Production Deployment Readiness & Specification Compliance Audit  

---

## Executive Summary & Readiness Verdict

**Overall Readiness Status**: **NOT READY FOR LINUX/EC2**  

While the core machine learning components (Config H blocking, 96-feature extraction, champion XGBoost model, and 0.72 threshold) are correctly specified and verified, the current deployment scripts contain critical execution blockers for Linux/EC2:
1. **Immediate Execution Failure**: `run_aws_inference.sh` attempts to install `requirements.txt` from the working directory root instead of `code/business_entity_resolution/requirements.txt`.
2. **Missing Dependency**: `psutil` is imported by `run_full_inference.py` (line 20) but missing from `code/business_entity_resolution/requirements.txt`, leading to an immediate `ModuleNotFoundError` in a clean environment.
3. **Data Loss / Silent Exclusion Risk**: `run_full_inference.py` (line 312) hardcodes a loop over `["France", "US", "India"]`. Any Source 1 entities from other countries in the test set will be silently ignored and omitted from inference output.
4. **Rigid Path & S3 Parameterization**: Dataset, model, metadata, and output directory paths cannot be customized via CLI arguments or environment variables, and `s3://` URI schemes are not supported natively.
5. **No Resiliency/Checkpointing**: Interrupted runs cannot be resumed; output files are overwritten with mode `"w"` on start.

---

## 1. Specification Verification Matrix

| Specification Item | Required Value / State | Actual Implementation State | Status |
| :--- | :--- | :--- | :---: |
| **Blocking Configuration** | Config H | Implements all 10 Config H blocking rules (`blocked_exact_name`, `blocked_compact_name`, `blocked_suffix_name`, `blocked_translit`, `blocked_postal`, `blocked_house`, `blocked_address_token`, `blocked_char_ngram`, `blocked_consonant_skeleton`, `blocked_address_name_combo`) | **PASS** |
| **Feature Count** | Exactly 96 features | `len(FEATURE_NAMES) == 96`, `NUM_FEATURES == 96`, `best_model.json` `num_feature == 96`, verified dynamically via assertion in `run_full_inference.py:272` | **PASS** |
| **Champion Model File** | `best_model.json` | Loaded via `MODEL_PATH` pointing to `code/business_entity_resolution/models/best_model.json` (`run_full_inference.py:49, 270`) | **PASS** |
| **Classification Threshold** | `tau = 0.72` | Loaded dynamically from `model_metadata.json` (`"selected_threshold": 0.72`) and applied at `run_full_inference.py:225, 263` | **PASS** |

---

## 2. Comprehensive Audit Check Matrix (17 Items)

| # | Check Item | Status | Problematic File & Line Number(s) | Impact / Root Cause | Recommended Fix |
| :-: | :--- | :-: | :--- | :--- | :--- |
| **1** | **Windows-specific paths (e.g. `C:\...`)** | **PASS** | N/A | Paths use POSIX format or `os.path.join` / `os.path.abspath`. No hardcoded backslashes or drive letters. | None required. |
| **2** | **Hardcoded absolute paths** | **PASS** | N/A | Base paths are calculated dynamically relative to `__file__` (`SRC_DIR` & `BASE_DIR` in `run_full_inference.py:27-28`). | None required. |
| **3** | **Assumptions about Current Working Directory (CWD)** | **FAIL** | [run_aws_inference.sh:18](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L18), [run_aws_inference.sh:23](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L23), [run_aws_inference.sh:31](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L31), [run_aws_inference.sh:40](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L40) | `run_aws_inference.sh` assumes it is executed from repository root and looks for `requirements.txt` at `./requirements.txt` instead of `code/business_entity_resolution/requirements.txt`. | Update `run_aws_inference.sh` line 18 to `python3 -m pip install -r code/business_entity_resolution/requirements.txt` and resolve repository paths relative to `SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)`. |
| **4** | **Hardcoded dataset locations** | **FAIL** | [run_full_inference.py:52-54](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L52-L54), [run_aws_inference.sh:34](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L34) | `TEST_S1_PATH`, `TEST_S2_PATH`, and `TEST_S3_PATH` are hardcoded module constants pointing to `BASE_DIR/dataset/test/test_source*.tsv`. `main()` lacks CLI flags for custom dataset paths. | Add CLI arguments `--test-s1-path`, `--test-s2-path`, `--test-s3-path`, and `--dataset-dir` to `run_full_inference.py` `argparse`. |
| **5** | **Hardcoded output locations** | **FAIL** | [run_aws_inference.sh:25-26](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L25-L26), [run_aws_inference.sh:32-33](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L32-L33), [run_aws_inference.sh:39-44](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L39-L44) | `run_aws_inference.sh` hardcodes `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and `output/submission.zip` without allowing output directory or paths to be configured via environment variables or CLI flags. | Parameterize `run_aws_inference.sh` with `OUTPUT_DIR="${OUTPUT_DIR:-output}"` and pass `$OUTPUT_DIR` dynamically to python calls and `zip`. |
| **6** | **Dependencies unavailable on standard Linux** | **PASS** *(Warning)* | [requirements.txt:1-7](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/requirements.txt#L1-L7), [run_full_inference.py:20](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L20) | All packages (`pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `lightgbm`) are available on Linux. However, `import psutil` in `run_full_inference.py:20` will crash standard Linux environments because `psutil` is missing from `requirements.txt`. | Add `psutil>=5.9.0` to `code/business_entity_resolution/requirements.txt`. |
| **7** | **Hardcoded S1 row counts** | **PASS** | N/A | Row count (1.73M) is only referenced in docstrings/comments (`run_full_inference.py:5`, `run_aws_inference.sh:22`). Ingestion loop dynamically calculates bounds (`len(s1_records)`). | None required. |
| **8** | **Hardcoded country counts / names** | **FAIL** | [run_full_inference.py:312](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L312) | Line 312 explicitly iterates over `for country in ["France", "US", "India"]:`! If the test set contains records from any other country (e.g. Germany, UK, Japan), those entities will be silently skipped. | Replace line 312 with `for country in sorted(s1_by_country.keys()):`. |
| **9** | **Hardcoded threshold values** | **PASS** | [run_full_inference.py:263](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L263), [model_metadata.json:8](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/models/model_metadata.json#L8) | Threshold `tau = 0.72` is dynamically loaded from `model_metadata.json` (`selected_threshold`). | Add optional `--threshold` argument to `argparse` to allow CLI overrides without mutating `model_metadata.json`. |
| **10** | **Hardcoded feature counts** | **PASS** | N/A | Feature count (96) is dynamically derived via `len(FEATURE_NAMES)` in `features.py` and asserted against `model_metadata.json` (`run_full_inference.py:272`). | None required. |
| **11** | **Hardcoded model paths** | **FAIL** | [run_full_inference.py:49-50](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L49-L50) | `MODEL_PATH` and `METADATA_PATH` are module-level constants. `argparse` does not expose `--model-path` or `--metadata-path` arguments. | Add `--model-path` and `--metadata-path` arguments to `argparse` in `run_full_inference.py`. |
| **12** | **Hardcoded test filenames** | **FAIL** | [run_full_inference.py:52-54](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L52-L54) | `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` filenames are fixed module constants without CLI parameterization. | Allow custom filenames via CLI arguments `--s1-name`, `--s2-name`, `--s3-name`. |
| **13** | **Assumptions that entire dataset fits in RAM** | **FAIL** | [run_full_inference.py:276-295](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L276-L295) | `run_full_inference.py` ingests all 1.73M Source 1 records into a single in-memory dictionary `s1_by_country`. On larger test sets (e.g. 10M+ entities), this will result in OOM. | Implement streaming S1 ingestion or disk-backed key-value store (e.g., SQLite / DuckDB / LMDB) for S1 lookup. |
| **14** | **Assumptions that entire candidate set fits in RAM** | **PASS** | N/A | Candidate generation, feature extraction, scoring, and file flushing are executed in bounded batches of 50,000 S1 records with `MAX_S1_CANDS=150` and 50,000-chunk feature matrices. | None required. |
| **15** | **Checkpoint / resume support** | **FAIL** | [run_full_inference.py:307-308](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L307-L308) | Output files are opened in `"w"` mode on startup, truncating any existing progress. No state file tracks completed countries/batches. | Add checkpoint file tracking completed partition batch IDs and open output files in `"a"` mode when resuming. |
| **16** | **Supply S3 paths cleanly** | **FAIL** | [run_full_inference.py:139](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L139), [run_full_inference.py:277](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L277), [run_aws_inference.sh:23-26](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L23-L26) | Python `open()` calls require local file paths. Neither `smart_open` / `boto3` integration nor `aws s3 cp` steps exist in `run_aws_inference.sh`. | Add S3 download/upload pre/post-processing steps to `run_aws_inference.sh` using `aws s3 sync` or `aws s3 cp`. |
| **17** | **Configure EC2 local filesystem paths cleanly** | **FAIL** | [run_aws_inference.sh:23-35](file:///c:/Users/nitya/Documents/student_resource/run_aws_inference.sh#L23-L35), [run_full_inference.py:49-58](file:///c:/Users/nitya/Documents/student_resource/code/business_entity_resolution/src/run_full_inference.py#L49-L58) | Paths for input datasets, models, and outputs cannot be passed cleanly via shell script environment variables or CLI flags for mounted EBS volumes (e.g. `/mnt/ebs/data`). | Add environment variable overrides (`DATA_DIR`, `MODEL_DIR`, `OUTPUT_DIR`) to `run_aws_inference.sh` and wire them to `run_full_inference.py` CLI arguments. |

---

## 3. Detailed Remediation Plan

To make the production pipeline fully ready for Linux/EC2 deployment without changing the model, Config H, or feature schema:

### A. Fix `code/business_entity_resolution/requirements.txt`
Add `psutil` to the requirements list:
```text
pandas>=2.0,<3
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
psutil>=5.9.0
```

### B. Fix `run_aws_inference.sh`
1. Correct the relative path to `requirements.txt` (`code/business_entity_resolution/requirements.txt`).
2. Support environment variables for paths (`DATA_DIR`, `MODEL_DIR`, `OUTPUT_DIR`).
3. Add robust `SCRIPT_DIR` resolution.

### C. Fix `code/business_entity_resolution/src/run_full_inference.py`
1. **Dynamic Country Loop**: Change line 312 from `for country in ["France", "US", "India"]:` to `for country in sorted(s1_by_country.keys()):`.
2. **CLI Parameterization**: Add CLI arguments `--test-s1-path`, `--test-s2-path`, `--test-s3-path`, `--model-path`, `--metadata-path`, and `--threshold`.
3. **Resiliency**: Implement basic partition checkpointing so interrupted runs skip already completed country partitions.
