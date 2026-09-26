# Strict Preflight Integrity Check & Audit Report

> **Stage 10: Strict Preflight Integrity Verification**  
> **Repository:** `ML Challenge 2026: Business Entity Resolution`  
> **Evaluated Outputs:** `output/preflight_matching_results.tsv`, `output/preflight_candidate_pairs.tsv`  
> **Sample Size:** 10,000 Test Source 1 Entities  
> **Target Universe:** 9,969,589 Target Entities (`test_source2.tsv` + `test_source3.tsv`)  
> **Overall Integrity Verdict:** **PASS**  
> **Timestamp:** 2026-09-26 04:50:06 UTC

---

## Executive Summary

Before transitioning to the full-scale AWS inference execution across all 1,732,545 test Source 1 entities, an exhaustive, deterministic integrity audit was conducted on the 10,000-S1 preflight submission outputs. Every predicted entity ID, candidate ID, delimiter, row count, and country label was cross-referenced directly against raw test files (`test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv`).

### Overall Audit Verdict: **PASS**

All verification rules, consistency invariants, and official competition submission criteria were **100% satisfied with zero defects, zero leakage, and zero non-existent IDs**.

---

## 1. Comprehensive Integrity Checklist Results

| # | Verification Rule | Verdict | Exact Counts & Details |
| :-: | :--- | :-: | :--- |
| 1 | **Matching TSV Header & Delimiter** | **PASS** | header: ['source1_entity_id', 'matched_entity_ids']<br>expected_header: ['source1_entity_id', 'matched_entity_ids']<br>malformed_rows_count: 0 |
| 2 | **Every S1 in Sample Appears Exactly Once (Matching TSV)** | **PASS** | total_rows: 10000<br>unique_s1_ids: 10000<br>duplicate_s1_rows: 0<br>missing_s1_entities: 0<br>unexpected_s1_entities: 0 |
| 3 | **No Duplicate IDs or Self-Matches (Matching TSV)** | **PASS** | intra_list_duplicates: 0<br>s1_self_matches: 0<br>empty_predictions_singletons: 6051<br>non_empty_predictions: 3949<br>total_match_pairs: 9687 |
| 4 | **Candidate TSV Header & Delimiter** | **PASS** | header: ['source1_entity_id', 'candidate_entity_ids']<br>expected_header: ['source1_entity_id', 'candidate_entity_ids']<br>malformed_rows_count: 0 |
| 5 | **Every S1 in Sample Appears Exactly Once (Candidate TSV)** | **PASS** | total_rows: 10000<br>unique_s1_ids: 10000<br>duplicate_s1_rows: 0<br>missing_s1_entities: 0<br>unexpected_s1_entities: 0 |
| 6 | **Candidate Prefix Safety & Intra-List Uniqueness** | **PASS** | intra_list_duplicates: 0<br>s1_self_matches: 0<br>wrong_prefix_ids: 0<br>total_candidate_pairs: 1377365<br>unique_target_entities: 612189 |
| 7 | **Every Predicted Match is a Subset of Candidates** | **PASS** | subset_violations_count: 0<br>sample_violations: [] |
| 8 | **Every Candidate & Match ID Exists in Test Sources** | **PASS** | total_unique_candidates_tested: 612189<br>found_in_test_source2_or_3: 612189<br>unfound_ids_count: 0<br>sample_unfound_ids: [] |
| 9 | **Strict Country Consistency (No Cross-Country Pairs)** | **PASS** | cross_country_violations_count: 0<br>sample_violations: [] |
| 10 | **Official Submission Validator (utils/validate_submission.py)** | **PASS** | exit_code: 0<br>output_verdict: PASS — no blocking issues found. Safe to submit. |

---

## 2. Invariant Breakdown & Key Findings

1. **S1 Entity Completeness & Uniqueness:**
   - **Sample Cohort:** Exactly **10,000 Source 1 entities** evaluated (India: **4,619**, US: **3,891**, France: **1,490**).
   - **Row Count:** Exactly **10,000 data rows** present in both `preflight_matching_results.tsv` and `preflight_candidate_pairs.tsv`.
   - **Uniqueness:** **0 duplicate S1 rows**; every S1 in the sample appears exactly once in both files.
   - **Coverage:** **0 missing S1 entities** and **0 unexpected S1 entities**.

2. **Ground-Truth ID Existence in Test Sources:**
   - **Unique Candidates Verified:** All **612,189 unique candidate target IDs** generated across the 10k cohort were cross-referenced against all 9,969,589 raw rows in `test_source2.tsv` (4,887,273) and `test_source3.tsv` (5,082,316).
   - **Unfound Candidate IDs:** **0**. Every candidate ID is verified to physically exist in `test_source2.tsv` or `test_source3.tsv`.
   - **Unfound Match IDs:** **0**. Every predicted match ID is verified to physically exist in `test_source2.tsv` or `test_source3.tsv`.

3. **Prefix Safety & Self-Match Elimination:**
   - **Candidate Prefix Invariant:** $100\%$ of candidate IDs match `S2-` or `S3-`.
   - **Self-Matches:** **0 S1- IDs** appear as candidate or match targets.
   - **Intra-List Deduplication:** **0 repeated IDs** inside any candidate list or match list.

4. **Subset Invariant:**
   - **Strict Hierarchy:** `Matches(S1) ⊆ Candidates(S1)` holds for all 10,000 entities.
   - **Subset Violations:** **0**. Exactly **0** predicted match IDs exist outside their respective candidate list.

5. **Country Consistency:**
   - **Cross-Country Candidates:** **0**. France S1 entities only match France targets; US only matches US; India only matches India.
   - **Cross-Country Matches:** **0**. Complete geographical consistency preserved.

6. **Format & Delimiter Compliance:**
   - Both files are standard **tab-separated values (.tsv)** with exact headers:
     - `source1_entity_id	matched_entity_ids`
     - `source1_entity_id	candidate_entity_ids`
   - Singletons are correctly formatted as `s1_id	` (empty string after tab).
   - Official validator (`utils/validate_submission.py`) returned exit code **0** (`PASS — no blocking issues found. Safe to submit`).

---

## 3. Preparation for Full AWS Execution

Because every strict preflight verification check has passed unconditionally, the repository is now certified and ready for full-scale AWS deployment:
- Model champion fixed: XGBoost 50k cohort (`models/best_model.json`).
- Decision threshold fixed: $	au = 0.72$ (`models/model_metadata.json`).
- Feature schema fixed: 96 handcrafted features (`features.py`).
- Blocking pipeline fixed: Configuration H (`blocking.py`).
- Streaming inference engine: Chunked streaming verified with constant memory footprint and linear scalability.

---

## 4. Final Verdict Confirmation

```
================================================================================
STRICT PREFLIGHT INTEGRITY AUDIT: PASS
================================================================================
Total S1 Checked         : 10,000 (100.0% present, 0 missing, 0 dupes)
Total Candidate Pairs    : 1,377,365 (All exist in test_source2/3, 0 missing)
Total Predicted Matches  : 9,687 (All subset of candidates, 0 missing)
Singletons (Empty Rows)  : 6,051 (Formatted with exact tab delimiter)
Cross-Country Violations : 0
Self-Match Violations    : 0
Validator Exit Code      : 0 (PASS)
Audit Duration           : 10.72s (0.18 min)
Ready for AWS Deployment : YES
================================================================================
```
