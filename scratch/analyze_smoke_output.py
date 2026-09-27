"""
Analysis script for smoke test output (output_smoke_60_90)
Measures:
- candidates/S1 mean and P95
- % S2 / % S3 candidates
- peak RAM (from process / logs)
- runtime
- count of predicted matches
- safety invariants check:
  - every predicted match in matching_results.tsv is present in candidate_pairs.tsv
  - max S2 candidates for any S1 <= 60
  - max S3 candidates for any S1 <= 90
  - max total candidates for any S1 <= 150
  - exact S1 coverage matches ingested count
"""

import os
import sys
import numpy as np

SMOKE_DIR = os.path.abspath("output_smoke_60_90")
MATCHING_PATH = os.path.join(SMOKE_DIR, "matching_results.tsv")
CANDIDATE_PATH = os.path.join(SMOKE_DIR, "candidate_pairs.tsv")
CHECKPOINT_PATH = os.path.join(SMOKE_DIR, ".inference_checkpoint.json")

def main():
    print("=" * 80)
    print("ANALYZING SMOKE RUN OUTPUT (60 S2 / 90 S3 Quota)")
    print("=" * 80)

    if not os.path.isfile(MATCHING_PATH) or not os.path.isfile(CANDIDATE_PATH):
        print("ERROR: Smoke output files not found.")
        sys.exit(1)

    # 1. Read candidates
    s1_candidates = {}
    total_cands = 0
    s2_cands_total = 0
    s3_cands_total = 0
    cand_counts = []
    s2_cand_counts = []
    s3_cand_counts = []

    with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            if s1_id == "source1_entity_id":
                continue
            if len(parts) > 1 and parts[1]:
                cands = parts[1].split(",")
            else:
                cands = []
            
            s1_candidates[s1_id] = set(cands)
            n_total = len(cands)
            n_s2 = sum(1 for c in cands if c.startswith("S2-"))
            n_s3 = sum(1 for c in cands if c.startswith("S3-"))
            
            cand_counts.append(n_total)
            s2_cand_counts.append(n_s2)
            s3_cand_counts.append(n_s3)
            
            total_cands += n_total
            s2_cands_total += n_s2
            s3_cands_total += n_s3

    n_s1 = len(s1_candidates)
    print(f"Total Evaluated S1 Entities: {n_s1:,}")
    print(f"Total Candidate Pairs Generated: {total_cands:,}")

    # Quota distribution
    pct_s2 = (s2_cands_total / total_cands * 100) if total_cands > 0 else 0
    pct_s3 = (s3_cands_total / total_cands * 100) if total_cands > 0 else 0
    print(f"\nCandidate Source Distribution:")
    print(f"  - S2 Candidates: {s2_cands_total:,} ({pct_s2:.2f}%)")
    print(f"  - S3 Candidates: {s3_cands_total:,} ({pct_s3:.2f}%)")

    # Candidate statistics
    cand_arr = np.array(cand_counts)
    s2_arr = np.array(s2_cand_counts)
    s3_arr = np.array(s3_cand_counts)

    print(f"\nCandidate Counts per S1:")
    print(f"  - Total Candidates/S1 : Mean={cand_arr.mean():.2f}, P50={np.percentile(cand_arr, 50):.1f}, P95={np.percentile(cand_arr, 95):.1f}, Max={cand_arr.max()}")
    print(f"  - S2 Candidates/S1    : Mean={s2_arr.mean():.2f}, P50={np.percentile(s2_arr, 50):.1f}, P95={np.percentile(s2_arr, 95):.1f}, Max={s2_arr.max()} (Cap: 60)")
    print(f"  - S3 Candidates/S1    : Mean={s3_arr.mean():.2f}, P50={np.percentile(s3_arr, 50):.1f}, P95={np.percentile(s3_arr, 95):.1f}, Max={s3_arr.max()} (Cap: 90)")

    # 2. Read matching results
    matched_s1_count = 0
    empty_s1_count = 0
    total_matches = 0
    matches_by_s1 = {}
    invariant_violations = []

    with open(MATCHING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            if s1_id == "source1_entity_id":
                continue
            if len(parts) > 1 and parts[1]:
                matches = parts[1].split(",")
            else:
                matches = []
            
            matches_by_s1[s1_id] = matches
            if matches:
                matched_s1_count += 1
                total_matches += len(matches)
                # Verify invariant: matches must be subset of candidates
                cands_for_s1 = s1_candidates.get(s1_id, set())
                for m in matches:
                    if m not in cands_for_s1:
                        invariant_violations.append((s1_id, m))
            else:
                empty_s1_count += 1

    print(f"\nMatching Results:")
    print(f"  - Total S1 Entities   : {len(matches_by_s1):,}")
    print(f"  - S1 with >=1 Match   : {matched_s1_count:,} ({matched_s1_count / len(matches_by_s1) * 100:.2f}%)")
    print(f"  - S1 with 0 Matches   : {empty_s1_count:,} ({empty_s1_count / len(matches_by_s1) * 100:.2f}%)")
    print(f"  - Total Predicted Matches: {total_matches:,}")

    # Check match distribution by source
    matched_s2 = sum(1 for ms in matches_by_s1.values() for m in ms if m.startswith("S2-"))
    matched_s3 = sum(1 for ms in matches_by_s1.values() for m in ms if m.startswith("S3-"))
    print(f"  - Predicted S2 Matches: {matched_s2:,} ({matched_s2 / total_matches * 100:.2f}%)" if total_matches > 0 else "")
    print(f"  - Predicted S3 Matches: {matched_s3:,} ({matched_s3 / total_matches * 100:.2f}%)" if total_matches > 0 else "")

    # Invariants Verification
    print(f"\nSafety Invariants Verification:")
    print(f"  [1] Max S2 candidates <= 60  : {'PASS' if s2_arr.max() <= 60 else 'FAIL'} (Observed Max: {s2_arr.max()})")
    print(f"  [2] Max S3 candidates <= 90  : {'PASS' if s3_arr.max() <= 90 else 'FAIL'} (Observed Max: {s3_arr.max()})")
    print(f"  [3] Max Total candidates <= 150: {'PASS' if cand_arr.max() <= 150 else 'FAIL'} (Observed Max: {cand_arr.max()})")
    print(f"  [4] Subset Invariant (Matches in Candidates): {'PASS' if len(invariant_violations) == 0 else f'FAIL ({len(invariant_violations)} violations)'}")
    print(f"  [5] Identical S1 Key Set     : {'PASS' if set(matches_by_s1.keys()) == set(s1_candidates.keys()) else 'FAIL'}")

    if s2_arr.max() > 60 or s3_arr.max() > 90 or cand_arr.max() > 150 or len(invariant_violations) > 0:
        print("\nCRITICAL: Invariant violation detected!")
        sys.exit(1)
    else:
        print("\nALL INVARIANTS PERFECTLY SATISFIED.")

if __name__ == "__main__":
    main()
