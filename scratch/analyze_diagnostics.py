"""
ML Challenge 2026: Comprehensive Production Diagnostics & Local Validation Analysis
Analyzes:
1. Current full production output (output/ and output/submission_snapshot_now/)
2. Local validation performance across thresholds & countries
3. Blocking recall & true match recovery
4. Structural integrity & schema compliance
5. Training vs Test distribution comparisons
"""
import os
import sys
import json
import csv
import time
from collections import defaultdict, Counter
import numpy as np
import xgboost as xgb

BASE_DIR = os.path.abspath(".")
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
sys.path.insert(0, SRC_DIR)

from preprocessing import has_indic_characters

def run_analysis():
    print("=" * 80)
    print("COMPREHENSIVE PRODUCTION DIAGNOSTICS & ERROR EXPLANATION")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # PART 1: CURRENT PRODUCTION OUTPUT & SUBMISSION SNAPSHOT ANALYSIS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 1: CURRENT PRODUCTION OUTPUT ANALYSIS")
    print("=" * 50)

    # 1.1 Ingest Test S1 Metadata (Entity ID -> Country, Script, Has Address)
    print("\n[1.1] Ingesting Test Source 1 Entities Metadata...")
    test_s1_meta = {}
    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            eid, name, addr, ctry = row[0], row[1], row[2], row[3]
            test_s1_meta[eid] = {
                "country": ctry,
                "has_address": bool(addr.strip()),
                "has_indic": has_indic_characters(name),
                "script": "Indic" if has_indic_characters(name) else "Latin"
            }
    total_test_s1 = len(test_s1_meta)
    print(f"  Total Test S1 Entities: {total_test_s1:,}")
    test_country_counts = Counter(v["country"] for v in test_s1_meta.values())
    for ctry, count in sorted(test_country_counts.items()):
        print(f"    - {ctry:10s}: {count:,} ({count/total_test_s1*100:.2f}%)")

    # 1.2 Checkpoint Information
    ckpt_path = "output/.inference_checkpoint.json"
    ckpt = {}
    if os.path.isfile(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
    print(f"\n[1.2] Current Checkpoint Status:")
    print(f"  Completed Batches: {len(ckpt.get('completed_batches', []))}")
    print(f"  Written S1 Count : {ckpt.get('written_s1_count', 0):,} ({ckpt.get('written_s1_count', 0)/total_test_s1*100:.2f}%)")
    print(f"  Last Batch       : {ckpt.get('last_batch')}")

    # 1.3 Analyze Active Production Output vs Submission Snapshot
    for label, m_path, c_path in [
        ("SUBMISSION SNAPSHOT (Leaderboard candidate)", "output/submission_snapshot_now/matching_results.tsv", "output/submission_snapshot_now/candidate_pairs.tsv"),
        ("COMMITTED PRODUCTION RUN (output/)", "output/matching_results.tsv", "output/candidate_pairs.tsv")
    ]:
        if not os.path.isfile(m_path):
            print(f"\n[1.3] {label}: File not found ({m_path})")
            continue

        print(f"\n[1.3] Analyzing {label}...")
        print(f"  Matching path : {m_path} ({os.path.getsize(m_path):,} bytes)")
        if os.path.isfile(c_path):
            print(f"  Candidate path: {c_path} ({os.path.getsize(c_path):,} bytes)")

        # Read Matching Results
        s1_matches = {}
        target_sources = Counter()
        matches_per_s1 = Counter()
        country_matched = Counter()
        country_total = Counter()
        script_matched = Counter()
        script_total = Counter()
        addr_matched = Counter()
        addr_total = Counter()

        with open(m_path, "r", encoding="utf-8") as f:
            header = next(f).rstrip("\r\n").split("\t")
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                s1_id = parts[0]
                m_str = parts[1] if len(parts) > 1 else ""
                m_list = [x.strip() for x in m_str.split(",") if x.strip()]
                s1_matches[s1_id] = m_list
                matches_per_s1[len(m_list)] += 1

                meta = test_s1_meta.get(s1_id, {"country": "Unknown", "has_address": False, "script": "Unknown"})
                ctry = meta["country"]
                script = meta["script"]
                has_addr = meta["has_address"]

                country_total[ctry] += 1
                script_total[script] += 1
                addr_total[has_addr] += 1

                if len(m_list) > 0:
                    country_matched[ctry] += 1
                    script_matched[script] += 1
                    addr_matched[has_addr] += 1
                    for tid in m_list:
                        if tid.startswith("S2-"):
                            target_sources["Source2"] += 1
                        elif tid.startswith("S3-"):
                            target_sources["Source3"] += 1
                        else:
                            target_sources["Other"] += 1

        total_rows = len(s1_matches)
        total_matched_s1 = sum(1 for m in s1_matches.values() if len(m) > 0)
        total_empty_s1 = total_rows - total_matched_s1

        print(f"  Total Processed S1 Rows : {total_rows:,} ({total_rows/total_test_s1*100:.2f}% of test set)")
        print(f"  Matched S1 Entities     : {total_matched_s1:,} ({total_matched_s1/total_rows*100:.2f}%)")
        print(f"  Empty S1 Entities       : {total_empty_s1:,} ({total_empty_s1/total_rows*100:.2f}%)")
        print(f"  Matches per S1 Dist     : {dict(sorted(matches_per_s1.items()))}")
        print(f"  Matched Target Sources  : {dict(target_sources)}")

        print(f"\n  Match Rate by Country:")
        for ctry in sorted(country_total.keys()):
            tot = country_total[ctry]
            mtch = country_matched[ctry]
            print(f"    - {ctry:10s}: {mtch:,} / {tot:,} ({mtch/tot*100:.2f}% matched, {100 - mtch/tot*100:.2f}% empty)")

        print(f"\n  Match Rate by Script:")
        for scr in sorted(script_total.keys()):
            tot = script_total[scr]
            mtch = script_matched[scr]
            print(f"    - {scr:10s}: {mtch:,} / {tot:,} ({mtch/tot*100:.2f}% matched)")

        print(f"\n  Match Rate by Address Presence:")
        for has_addr in [True, False]:
            tot = addr_total[has_addr]
            mtch = addr_matched[has_addr]
            label_a = "Address Present" if has_addr else "Address Missing"
            if tot > 0:
                print(f"    - {label_a:17s}: {mtch:,} / {tot:,} ({mtch/tot*100:.2f}% matched)")
            else:
                print(f"    - {label_a:17s}: 0 / 0 (N/A)")

        # Candidate pairs distribution if candidate file exists
        if os.path.isfile(c_path):
            cand_counts = []
            sample_count = 0
            with open(c_path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    c_str = parts[1].strip() if len(parts) > 1 else ""
                    n_c = (c_str.count(",") + 1) if c_str else 0
                    cand_counts.append(n_c)
                    sample_count += 1
                    if sample_count > 500000 and label != "COMMITTED PRODUCTION RUN (output/)":
                        # sample if too large for memory
                        pass

            c_arr = np.array(cand_counts, dtype=np.int32)
            non_zero_cands = c_arr[c_arr > 0]
            print(f"\n  Candidate Count per S1 Distribution (over {len(c_arr):,} entities):")
            print(f"    - Entities with 0 cands : {(c_arr == 0).sum():,} ({(c_arr == 0).mean()*100:.2f}%)")
            print(f"    - Mean candidates/S1    : {c_arr.mean():.2f}")
            if len(non_zero_cands) > 0:
                print(f"    - Non-zero Min / P25 / P50 / P75 / P95 / P99 / Max:")
                print(f"      {np.min(non_zero_cands)} / {np.percentile(non_zero_cands, 25):.0f} / {np.percentile(non_zero_cands, 50):.0f} / {np.percentile(non_zero_cands, 75):.0f} / {np.percentile(non_zero_cands, 95):.0f} / {np.percentile(non_zero_cands, 99):.0f} / {np.max(non_zero_cands)}")

    # --------------------------------------------------------------------------
    # PART 2: VALIDATION / CHAMPION MODEL RE-EVALUATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 2: LOCAL VALIDATION RE-EVALUATION ACROSS THRESHOLDS & COUNTRIES")
    print("=" * 50)

    val_npz = np.load("dataset/processed/val_data.npz", allow_pickle=True)
    X_val = val_npz["X"]
    y_val = val_npz["y"]
    s1_val = val_npz["s1_ids"]
    cand_val = val_npz["cand_ids"]
    feat_names = list(val_npz["feature_names"])

    model = xgb.XGBClassifier()
    model.load_model("code/business_entity_resolution/models/best_model.json")

    # Load Ground Truth
    unique_s1 = sorted(list(set(s1_val)))
    val_gt = {s: set() for s in unique_s1}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in val_gt:
                val_gt[row[0]] = set(x.strip() for x in row[1].split(",") if x.strip())

    # Get Country for each validation S1
    val_s1_country = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in val_gt:
                val_s1_country[row[0]] = row[3]

    y_prob = model.predict_proba(X_val)[:, 1]

    def compute_metrics_for_subset(s1_subset, tau):
        s1_set = set(s1_subset)
        mask = (y_prob >= tau)
        s1_preds = defaultdict(set)
        for s, c, m in zip(s1_val, cand_val, mask):
            if m and s in s1_set:
                s1_preds[s].add(c)

        s1_scores = []
        correct_sing = 0
        total_sing = 0
        tp_pairs = 0
        fp_pairs = 0
        fn_pairs = 0

        for s in s1_subset:
            preds = s1_preds.get(s, set())
            truth = val_gt[s]

            # Entity F0.5
            if len(truth) == 0:
                score = 1.0 if len(preds) == 0 else 0.0
                total_sing += 1
                if len(preds) == 0:
                    correct_sing += 1
                else:
                    fp_pairs += len(preds)
            elif len(preds) == 0:
                score = 0.0
                fn_pairs += len(truth)
            else:
                tp = len(preds & truth)
                fp = len(preds - truth)
                fn = len(truth - preds)
                tp_pairs += tp
                fp_pairs += fp
                fn_pairs += fn
                if tp == 0:
                    score = 0.0
                else:
                    prec = tp / len(preds)
                    rec = tp / len(truth)
                    denom = (0.25 * prec) + rec
                    score = (1.25 * prec * rec) / denom if denom > 0 else 0.0

            s1_scores.append(score)

        macro_f05 = float(np.mean(s1_scores))
        sing_acc = (correct_sing / total_sing) if total_sing > 0 else 1.0
        prec = tp_pairs / (tp_pairs + fp_pairs) if (tp_pairs + fp_pairs) > 0 else 0.0
        rec = tp_pairs / (tp_pairs + fn_pairs) if (tp_pairs + fn_pairs) > 0 else 0.0

        return {
            "threshold": tau,
            "macro_f05": macro_f05,
            "precision": prec,
            "recall": rec,
            "singleton_accuracy": sing_acc,
            "tp": tp_pairs,
            "fp": fp_pairs,
            "fn": fn_pairs
        }

    # Threshold curve
    thresholds = [0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85]
    print(f"\n[2.1] Overall Validation Threshold Curve (N={len(unique_s1):,} S1 Entities):")
    print(f"  {'Tau':>6} | {'Macro F0.5':>10} | {'Precision':>10} | {'Recall':>10} | {'Sing Acc':>10} | {'FP':>6} | {'FN':>6}")
    print("  " + "-" * 72)
    for tau in thresholds:
        m = compute_metrics_for_subset(unique_s1, tau)
        marker = " <-- CHAMPION" if abs(tau - 0.72) < 1e-4 else ""
        print(f"  {m['threshold']:6.2f} | {m['macro_f05']*100:9.2f}% | {m['precision']*100:9.2f}% | {m['recall']*100:9.2f}% | {m['singleton_accuracy']*100:9.2f}% | {m['fp']:6d} | {m['fn']:6d}{marker}")

    # By Country
    us_s1 = [s for s in unique_s1 if val_s1_country.get(s) == "US"]
    ind_s1 = [s for s in unique_s1 if val_s1_country.get(s) == "India"]

    print(f"\n[2.2] India Subset Validation Curve (N={len(ind_s1):,} S1 Entities):")
    print(f"  {'Tau':>6} | {'Macro F0.5':>10} | {'Precision':>10} | {'Recall':>10} | {'Sing Acc':>10} | {'FP':>6} | {'FN':>6}")
    print("  " + "-" * 72)
    for tau in [0.65, 0.70, 0.72, 0.75, 0.80]:
        m = compute_metrics_for_subset(ind_s1, tau)
        print(f"  {m['threshold']:6.2f} | {m['macro_f05']*100:9.2f}% | {m['precision']*100:9.2f}% | {m['recall']*100:9.2f}% | {m['singleton_accuracy']*100:9.2f}% | {m['fp']:6d} | {m['fn']:6d}")

    print(f"\n[2.3] US Subset Validation Curve (N={len(us_s1):,} S1 Entities):")
    print(f"  {'Tau':>6} | {'Macro F0.5':>10} | {'Precision':>10} | {'Recall':>10} | {'Sing Acc':>10} | {'FP':>6} | {'FN':>6}")
    print("  " + "-" * 72)
    for tau in [0.65, 0.70, 0.72, 0.75, 0.80]:
        m = compute_metrics_for_subset(us_s1, tau)
        print(f"  {m['threshold']:6.2f} | {m['macro_f05']*100:9.2f}% | {m['precision']*100:9.2f}% | {m['recall']*100:9.2f}% | {m['singleton_accuracy']*100:9.2f}% | {m['fp']:6d} | {m['fn']:6d}")

    # --------------------------------------------------------------------------
    # PART 3: BLOCKING RECALL & TRAINING GROUND TRUTH ANALYSIS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 3: BLOCKING RECALL & GROUND TRUTH PATTERNS")
    print("=" * 50)

    # In validation set:
    total_val_gt_links = sum(len(m) for m in val_gt.values())
    recalled_in_val_cands = int((y_val == 1).sum())
    val_cand_pairs = set(zip(s1_val, cand_val))

    missed_val_links = []
    for s, targets in val_gt.items():
        for t in targets:
            if (s, t) not in val_cand_pairs:
                missed_val_links.append((s, t))

    print(f"  Validation Total Ground Truth Links: {total_val_gt_links:,}")
    print(f"  Candidate Set Recovered Links      : {recalled_in_val_cands:,}")
    print(f"  Validation Blocking Candidate Recall: {recalled_in_val_cands / total_val_gt_links * 100:.2f}%")
    print(f"  Total Blocking Misses (Lost at Gate): {len(missed_val_links):,} ({len(missed_val_links)/total_val_gt_links*100:.2f}%)")

    # Check Training Ground Truth Global Match Rate
    print("\n[3.2] Training Set Ground Truth Global Linkage Distribution:")
    train_gt_sizes = Counter()
    total_train_gt_links = 0
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            targets = [x.strip() for x in row[1].split(",") if x.strip()]
            train_gt_sizes[len(targets)] += 1
            total_train_gt_links += len(targets)

    total_train_s1 = sum(train_gt_sizes.values())
    train_matched_s1 = total_train_s1 - train_gt_sizes[0]
    print(f"  Total Training S1 Entities     : {total_train_s1:,}")
    print(f"  Training Singletons (0 matches): {train_gt_sizes[0]:,} ({train_gt_sizes[0]/total_train_s1*100:.2f}%)")
    print(f"  Training Non-Singletons (>=1)  : {train_matched_s1:,} ({train_matched_s1/total_train_s1*100:.2f}%)")
    print(f"  Matches per S1 Breakdown       : {dict(sorted(train_gt_sizes.items()))}")
    print(f"  Mean Matches per Non-Singleton : {total_train_gt_links / train_matched_s1:.3f}")

    # --------------------------------------------------------------------------
    # PART 4: STRUCTURAL VALIDATION CHECKS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 4: STRUCTURAL INTEGRITY VALIDATION")
    print("=" * 50)

    # Validate output/submission_snapshot_now/matching_results.tsv
    snap_m_path = "output/submission_snapshot_now/matching_results.tsv"
    snap_c_path = "output/submission_snapshot_now/candidate_pairs.tsv"

    seen_s1 = set()
    dup_s1 = 0
    invalid_format = 0
    self_matches = 0
    m_not_in_c = 0

    with open(snap_m_path, "r", encoding="utf-8") as fm, open(snap_c_path, "r", encoding="utf-8") as fc:
        h_m = next(fm).strip()
        h_c = next(fc).strip()
        assert h_m == "source1_entity_id\tmatched_entity_ids"
        assert h_c == "source1_entity_id\tcandidate_entity_ids"

        for line_m, line_c in zip(fm, fc):
            s1_m, tab_m, m_str = line_m.partition("\t")
            s1_c, tab_c, c_str = line_c.partition("\t")

            if s1_m in seen_s1:
                dup_s1 += 1
            seen_s1.add(s1_m)

            if s1_m != s1_c:
                invalid_format += 1

            m_set = set(x.strip() for x in m_str.rstrip("\r\n").split(",") if x.strip())
            c_set = set(x.strip() for x in c_str.rstrip("\r\n").split(",") if x.strip())

            for m in m_set:
                if m.startswith("S1-"):
                    self_matches += 1
                if m not in c_set:
                    m_not_in_c += 1

    print(f"  Total Validated S1 Rows        : {len(seen_s1):,}")
    print(f"  Exact Coverage (vs 1,732,544)  : {len(seen_s1) == 1732544}")
    print(f"  Duplicate S1 Rows              : {dup_s1}")
    print(f"  Alignment Mismatches (M vs C)  : {invalid_format}")
    print(f"  Self-Matches (S1 in matched)   : {self_matches}")
    print(f"  Matches Absent from Candidates : {m_not_in_c}")

    # --------------------------------------------------------------------------
    # PART 5: COMPARATIVE DIAGNOSTICS & THE "SMOKING GUN"
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PART 5: COMPARATIVE DIAGNOSTIC & DISTRIBUTION SHIFTS")
    print("=" * 50)

    train_match_rate = train_matched_s1 / total_train_s1
    test_snapshot_match_rate = total_matched_s1 / total_test_s1

    print(f"  [D1] GLOBAL MATCH RATE COMPARISON:")
    print(f"    - Training Ground Truth Match Rate   : {train_match_rate*100:.2f}% (Non-Singletons)")
    print(f"    - Validation Model Predicted Rate    : {(y_val == 1).sum() / len(unique_s1):.2f}%")
    print(f"    - Full Test Submission Snapshot Rate : {test_snapshot_match_rate*100:.2f}% (Active on Leaderboard)")
    print(f"    - Active France Predicted Match Rate : {country_matched['France'] / country_total['France']*100:.2f}%")
    print(f"    - Active India Predicted Match Rate  : {country_matched['India'] / country_total['India']*100:.2f}%")
    print(f"    - Active US Predicted Match Rate     : {country_matched.get('US', 0) / country_total.get('US', 1)*100:.2f}%")

    print(f"\n  [D2] EMPTY-PREDICTION / SINGLETON ANOMALY:")
    print(f"    - Expected Empty Rate from Training : {train_gt_sizes[0]/total_train_s1*100:.2f}%")
    print(f"    - France Empty Rate in Prediction   : {100 - country_matched['France'] / country_total['France']*100:.2f}%")
    print(f"    - India Empty Rate in Prediction    : {100 - country_matched['India'] / country_total['India']*100:.2f}%")
    print(f"    - US Empty Rate in Current Snapshot : {100 - country_matched.get('US', 0) / country_total.get('US', 1)*100:.2f}% (PADDED!)")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_analysis()
