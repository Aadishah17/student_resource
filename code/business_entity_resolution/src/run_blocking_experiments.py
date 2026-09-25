"""
ML Challenge 2026: Business Entity Resolution
Module: run_blocking_experiments.py

Runs the complete empirical validation and ablation experiments for Candidate Generation (Blocking).
Evaluates individual blocking rules and combination pipelines (A through G) across:
- Overall candidate recall
- Subgroup recalls (US, India, Latin->Latin, Latin->Indic cross-script)
- Candidate volume statistics (Mean, Median, P95, Max)
- Search space reduction ratio
- Computational runtime & memory profiling
"""

import os
import sys
import time
import csv
import json
import random
import tracemalloc
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

# Import modules
from preprocessing import (
    normalize_name,
    remove_legal_suffix,
    compact_string,
    transliterate_name,
    normalize_address,
    extract_postal_code,
    extract_house_number,
    has_indic_characters
)
from blocking import (
    InvertedIndexBlocker,
    NgramBlocker,
    CandidateGenerator,
    ADDR_STOPWORDS
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def run_experiments():
    print("=" * 80)
    print("RUNNING CANDIDATE GENERATION (BLOCKING) EMPIRICAL VALIDATION")
    print("=" * 80)

    tracemalloc.start()
    t_global_start = time.time()

    # 1. Stratified S1 Cohort Selection (10,000 entities: 6,000 US, 4,000 India)
    print("\n[Phase 1] Selecting Stratified Validation Cohort...")
    random.seed(42)
    s1_cohort: Dict[str, Dict[str, Any]] = {}
    us_count, ind_count = 0, 0
    us_target, ind_target = 6000, 4000

    s1_path = os.path.join(BASE_DIR, "dataset", "train", "train_source1.tsv")
    with open(s1_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            if country == "US" and us_count < us_target:
                norm_n = normalize_name(name)
                s1_cohort[eid] = {
                    "entity_id": eid, "name": name, "address": addr, "country": country,
                    "norm_name": norm_n, "compact_name": compact_string(norm_n),
                    "nosuff_name": remove_legal_suffix(norm_n),
                    "translit_name": transliterate_name(name),
                    "postal_code": extract_postal_code(addr),
                    "house_number": extract_house_number(addr),
                    "norm_address": normalize_address(addr),
                }
                us_count += 1
            elif country == "India" and ind_count < ind_target:
                norm_n = normalize_name(name)
                s1_cohort[eid] = {
                    "entity_id": eid, "name": name, "address": addr, "country": country,
                    "norm_name": norm_n, "compact_name": compact_string(norm_n),
                    "nosuff_name": remove_legal_suffix(norm_n),
                    "translit_name": transliterate_name(name),
                    "postal_code": extract_postal_code(addr),
                    "house_number": extract_house_number(addr),
                    "norm_address": normalize_address(addr),
                }
                ind_count += 1
            if us_count >= us_target and ind_count >= ind_target:
                break

    print(f"  Selected: {len(s1_cohort):,} S1 entities ({us_count:,} US, {ind_count:,} India)")

    # 2. Ground Truth Extraction
    print("\n[Phase 2] Loading Ground Truth Labels...")
    ground_truth: Dict[str, Set[str]] = {}
    target_ids_needed = set()
    gt_path = os.path.join(BASE_DIR, "dataset", "train", "train_ground_truth.tsv")

    with open(gt_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            s1_id = row[0]
            if s1_id in s1_cohort:
                matches = set(x.strip() for x in row[1].split(",") if x.strip())
                ground_truth[s1_id] = matches
                target_ids_needed.update(matches)

    total_true_links = sum(len(m) for m in ground_truth.values())
    singletons = sum(1 for m in ground_truth.values() if len(m) == 0)
    print(f"  Total True Links: {total_true_links:,}")
    print(f"  Singletons: {singletons:,} ({singletons / len(s1_cohort) * 100:.2f}%)")
    print(f"  Target True Records Needed: {len(target_ids_needed):,}")

    # 3. Load Target Records (True Matches + 500k Distractors)
    print("\n[Phase 3] Ingesting Target Pool (True Matches + 500,000 Distractors)...")
    target_records: Dict[str, Dict[str, Any]] = {}
    needed_s2 = {eid for eid in target_ids_needed if eid.startswith("S2-")}
    needed_s3 = {eid for eid in target_ids_needed if eid.startswith("S3-")}

    for fname, needed_set, distractor_limit in [
        ("train_source2.tsv", needed_s2, 250000),
        ("train_source3.tsv", needed_s3, 250000)
    ]:
        fpath = os.path.join(BASE_DIR, "dataset", "train", fname)
        distractors_added = 0
        with open(fpath, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for row in r:
                eid, name, addr, country = row[0], row[1], row[2], row[3]
                is_needed = eid in needed_set
                if is_needed or distractors_added < distractor_limit:
                    norm_n = normalize_name(name)
                    has_ind = has_indic_characters(name)
                    trans_n = transliterate_name(name) if has_ind else norm_n
                    target_records[eid] = {
                        "entity_id": eid, "name": name, "address": addr, "country": country,
                        "norm_name": norm_n, "compact_name": compact_string(norm_n),
                        "nosuff_name": remove_legal_suffix(norm_n),
                        "translit_name": trans_n,
                        "has_indic": has_ind,
                        "postal_code": extract_postal_code(addr),
                        "house_number": extract_house_number(addr),
                        "norm_address": normalize_address(addr),
                    }
                    if not is_needed:
                        distractors_added += 1

    print(f"  Target Records Ingested: {len(target_records):,}")

    # Track Subgroup Links
    true_links_us = set()
    true_links_ind = set()
    true_links_latin = set()
    true_links_cross = set()

    for s1_id, matches in ground_truth.items():
        country = s1_cohort[s1_id]["country"]
        for m in matches:
            if country == "US":
                true_links_us.add((s1_id, m))
            else:
                true_links_ind.add((s1_id, m))
            if m in target_records:
                if target_records[m]["has_indic"]:
                    true_links_cross.add((s1_id, m))
                else:
                    true_links_latin.add((s1_id, m))

    print(f"  Ground Truth Categories: US={len(true_links_us):,}, India={len(true_links_ind):,}, Latin-Latin={len(true_links_latin):,}, Cross-Script={len(true_links_cross):,}")

    # 4. Construct Inverted Indexes
    print("\n[Phase 4] Constructing Inverted Indexes & High-Frequency Bucket Profiling...")
    t_idx_start = time.time()
    blocker = InvertedIndexBlocker(max_bucket_size=500)
    target_list = list(target_records.values())
    blocker.compute_address_token_frequencies(target_list)
    for rec in target_list:
        blocker.add_record(rec)

    idx_time = time.time() - t_idx_start
    print(f"  Inverted Indexes built in {idx_time:.2f}s.")

    # High-frequency bucket profiling
    high_freq_keys = []
    for country in blocker.indexes:
        for rule in blocker.indexes[country]:
            for key, bucket in blocker.indexes[country][rule].items():
                if len(bucket) >= 150:
                    high_freq_keys.append((country, rule, str(key)[:40], len(bucket)))
    high_freq_keys.sort(key=lambda x: x[3], reverse=True)

    print(f"  High-frequency buckets (>=150 records): {len(high_freq_keys)}")
    for item in high_freq_keys[:8]:
        print(f"    Country: {item[0]:<6} | Rule: {item[1]:<14} | Key: {item[2]:<30} | Size: {item[3]:,}")

    # 5. Fit Character N-Gram Nearest Neighbor Models
    print("\n[Phase 5] Training Character N-Gram Nearest-Neighbor Models...")
    t_ngram_start = time.time()
    ngram_blocker = NgramBlocker(ngram_range=(3, 5), analyzer="char_wb", min_df=2, max_df=0.25, top_k=10, min_similarity=0.40)
    by_country = defaultdict(list)
    for r in target_list:
        by_country[r["country"]].append(r)
    for country, recs in by_country.items():
        ngram_blocker.fit_target_records(country, recs)

    s1_list_us = [r for r in s1_cohort.values() if r["country"] == "US"]
    s1_list_ind = [r for r in s1_cohort.values() if r["country"] == "India"]

    ngram_cands = {
        **ngram_blocker.query_batch("US", s1_list_us),
        **ngram_blocker.query_batch("India", s1_list_ind)
    }
    ngram_time = time.time() - t_ngram_start
    print(f"  N-Gram Models fitted and queried in {ngram_time:.2f}s.")

    # 6. Evaluation Routine
    cartesian_space = len(s1_cohort) * len(target_records)

    def evaluate_strategy(cands_map: Dict[str, Set[str]], name: str) -> Dict[str, Any]:
        counts = [len(cands_map.get(s1_id, set())) for s1_id in s1_cohort]
        found_total, found_us, found_ind, found_lat, found_cross = 0, 0, 0, 0, 0
        for s1_id, cset in cands_map.items():
            for m in ground_truth.get(s1_id, set()):
                if m in cset:
                    found_total += 1
                    if (s1_id, m) in true_links_us: found_us += 1
                    if (s1_id, m) in true_links_ind: found_ind += 1
                    if (s1_id, m) in true_links_latin: found_lat += 1
                    if (s1_id, m) in true_links_cross: found_cross += 1

        total_pairs = sum(counts)
        red_ratio = 1.0 - (total_pairs / cartesian_space) if cartesian_space > 0 else 1.0

        return {
            "name": name,
            "recall_total": found_total / total_true_links,
            "recall_us": found_us / len(true_links_us),
            "recall_ind": found_ind / len(true_links_ind),
            "recall_latin": found_lat / len(true_links_latin),
            "recall_cross": found_cross / len(true_links_cross),
            "mean_cands": float(np.mean(counts)),
            "median_cands": float(np.median(counts)),
            "p95_cands": float(np.percentile(counts, 95)),
            "max_cands": int(np.max(counts)) if counts else 0,
            "total_pairs": total_pairs,
            "reduction_ratio": red_ratio
        }

    # 7. Individual Rule Evaluations
    print("\n[Phase 6] Evaluating Individual Blocking Rules...")
    individual_results = []
    rule_sets = [
        ("Exact Normalized Name Only", {"exact"}),
        ("Transliteration Keys Only", {"translit"}),
        ("Postal Combined Keys Only", {"postal"}),
        ("House Number Keys Only", {"house"}),
        ("Address Token Keys Only", {"address"}),
    ]

    for label, r_set in rule_sets:
        c_map = {eid: blocker.query(r, r_set) for eid, r in s1_cohort.items()}
        res = evaluate_strategy(c_map, label)
        individual_results.append(res)
        print(f"  {label:<32} | Recall: {res['recall_total']*100:5.2f}% | Cross: {res['recall_cross']*100:5.2f}% | Avg: {res['mean_cands']:5.2f} | P95: {res['p95_cands']:4.1f} | Pairs: {res['total_pairs']:,}")

    # N-gram standalone
    res_ngram = evaluate_strategy(ngram_cands, "Character N-Gram Nearest-Neighbor Only")
    individual_results.append(res_ngram)
    print(f"  {res_ngram['name']:<32} | Recall: {res_ngram['recall_total']*100:5.2f}% | Cross: {res_ngram['recall_cross']*100:5.2f}% | Avg: {res_ngram['mean_cands']:5.2f} | P95: {res_ngram['p95_cands']:4.1f} | Pairs: {res_ngram['total_pairs']:,}")

    # 8. Cumulative Combination Ablation Experiments (A through G)
    print("\n[Phase 7] Evaluating Cumulative Combination Pipeline (A through G)...")
    ablation_results = []

    configs = [
        ("A. Exact-Name Only", {"exact"}, False),
        ("B. Exact + Postal", {"exact", "postal"}, False),
        ("C. Exact + Postal + House Number", {"exact", "postal", "house"}, False),
        ("D. Add Transliteration", {"exact", "postal", "house", "translit"}, False),
        ("E. Add Character N-Gram", {"exact", "postal", "house", "translit"}, True),
        ("F. Add Address Retrieval (No N-Gram)", {"exact", "postal", "house", "translit", "address"}, False),
        ("G. Full Union (All Rules + N-Gram)", {"exact", "postal", "house", "translit", "address"}, True),
    ]

    for label, r_set, inc_ngram in configs:
        t0 = time.time()
        c_map = {}
        for eid, r in s1_cohort.items():
            s = blocker.query(r, r_set)
            if inc_ngram:
                s = s | ngram_cands.get(eid, set())
            c_map[eid] = s
        el = time.time() - t0

        res = evaluate_strategy(c_map, label)
        res["runtime_sec"] = el
        ablation_results.append(res)

        print(f"\n[{label}] (Query time: {el:.2f}s)")
        print(f"  Recall Total : {res['recall_total']*100:6.2f}% | US: {res['recall_us']*100:6.2f}% | India: {res['recall_ind']*100:6.2f}%")
        print(f"  Latin-Latin  : {res['recall_latin']*100:6.2f}% | Cross-Script: {res['recall_cross']*100:6.2f}%")
        print(f"  Candidates/S1: Mean={res['mean_cands']:5.2f}, Median={res['median_cands']:4.1f}, P95={res['p95_cands']:4.1f}, Max={res['max_cands']}")
        print(f"  Pairs        : {res['total_pairs']:,} (Search space reduction: {res['reduction_ratio']*100:9.6f}%)")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_runtime = time.time() - t_global_start

    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Runtime: {total_runtime:.2f} seconds")
    print(f"Peak Memory  : {peak_mem / (1024 * 1024):.2f} MB")
    print(f"Cartesian Search Space: {cartesian_space:,} pairs")

    return {
        "individual_results": individual_results,
        "ablation_results": ablation_results,
        "high_freq_keys": high_freq_keys,
        "runtime_sec": total_runtime,
        "peak_mem_mb": peak_mem / (1024 * 1024),
        "cartesian_space": cartesian_space
    }


if __name__ == "__main__":
    results = run_experiments()
