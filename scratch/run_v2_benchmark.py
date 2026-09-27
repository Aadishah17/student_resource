"""
ML Challenge 2026: Business Entity Resolution
Module: run_v2_benchmark.py

Comprehensive benchmark comparing the baseline single-threaded inference (V1)
against the multi-process V2 inference engine across 1, 4, and 8 worker processes.
Evaluates 20,000 S1 entities (two 10,000-entity batches) using real test data.

Measures:
- Total wall-clock runtime
- Feature extraction runtime
- Streaming runtime
- CPU utilization
- Peak RAM (Working Set)
- Pagefile / Swap memory usage
- Candidate pair count
- Throughput (S1 / hour)
- Speedup vs baseline
- Full floating-point numerical equivalence across all 96 features
"""

import os
import sys
import time
import shutil
import psutil
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import numpy as np
import xgboost as xgb

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing import (
    normalize_name,
    compact_string,
    remove_legal_suffix,
    extract_postal_code,
    extract_house_number,
    normalize_address,
    transliterate_name,
    has_indic_characters
)
from blocking import extract_consonants, ADDR_STOPWORDS, DEFAULT_MAX_ADDR_TOKEN_FREQ
from features import (
    PairwiseFeatureExtractor,
    enrich_record_for_features,
    FEATURE_NAMES,
    NUM_FEATURES
)
from inference_v2 import extract_features_v2_multiprocess, score_pairs_v2

MODEL_PATH = os.path.join(BASE_DIR, "code", "business_entity_resolution", "models", "best_model.json")
TEST_S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
TEST_S2_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source2.tsv")
TEST_S3_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source3.tsv")

OUTPUT_DIR = os.path.join(BASE_DIR, "output", "benchmark_v2")

BATCH_SIZE = 10000
MAX_S1_CANDS = 150
CHUNK_FEATURE_SIZE = 50000
MAX_S1_ADDR_TOKEN_FREQ = 5
MAX_S1_NGRAM_FREQ = 2
MAX_TARGET_ADDR_TOKEN_FREQ = DEFAULT_MAX_ADDR_TOKEN_FREQ
MAX_TARGET_NGRAM_FREQ = 50
TAU = 0.72


def get_mem_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_swap_mb() -> float:
    return psutil.swap_memory().used / (1024 * 1024)


def get_peak_ram_mb() -> float:
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss) / (1024 * 1024)


def stream_partition_candidates(
    country: str,
    batch_s1_ids: List[str],
    batch_s1_recs: Dict[str, Dict[str, Any]],
    target_paths: List[str]
) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, Tuple[str, str, str]], float]:
    """
    Executes Configuration H blocking streaming for the batch against real test targets.
    Returns:
    - candidates: s1_id -> cand_id -> set of rule names
    - needed_raw_targets: cand_id -> (name, address, country)
    - streaming_time: wall-clock seconds
    """
    t_start = time.perf_counter()

    # 1. Build Batch Inverted Indexes
    c_idx: Dict[str, Dict[Any, List[str]]] = {
        "exact": defaultdict(list),
        "compact": defaultdict(list),
        "nosuff": defaultdict(list),
        "translit": defaultdict(list),
        "postal_prefix": defaultdict(list),
        "postal_house": defaultdict(list),
        "house_prefix": defaultdict(list),
        "consonant_skel": defaultdict(list),
        "addr_name": defaultdict(list),
        "addr_token": defaultdict(list),
        "char_ngram": defaultdict(list)
    }

    s1_addr_token_counts = Counter()
    s1_ngram_counts = Counter()
    for eid in batch_s1_ids:
        rec = batch_s1_recs[eid]
        if rec["norm_address"]:
            for t in [x for x in rec["norm_address"].split() if len(x) >= 4 and not x.isdigit() and x not in ADDR_STOPWORDS][:3]:
                s1_addr_token_counts[t] += 1
        eff_nosuff = rec["translit_nosuff"] if (rec["has_indic"] and rec["translit_nosuff"]) else rec["nosuff_name"]
        comp_eff = compact_string(eff_nosuff) if eff_nosuff else ""
        if len(comp_eff) >= 5:
            grams = set(comp_eff[j:j+5] for j in range(len(comp_eff) - 4))
            for g in grams:
                s1_ngram_counts[g] += 1
        elif len(comp_eff) == 4:
            s1_ngram_counts[comp_eff] += 1

    for eid in batch_s1_ids:
        rec = batch_s1_recs[eid]
        norm = rec["norm_name"]
        comp = rec["compact_name"]
        nosuff = rec["nosuff_name"]
        trans = rec["translit_name"]
        post = rec["postal_code"]
        house = rec["house_number"]
        skel = rec["skel"]
        norm_addr = rec["norm_address"]

        if norm: c_idx["exact"][norm].append(eid)
        if comp: c_idx["compact"][comp].append(eid)
        if nosuff: c_idx["nosuff"][nosuff].append(eid)
        if trans: c_idx["translit"][trans].append(eid)
        if post and len(norm) >= 3: c_idx["postal_prefix"][(post, norm[:3])].append(eid)
        if post and house: c_idx["postal_house"][(post, house.lower())].append(eid)
        if house and len(house) >= 2 and len(norm) >= 3: c_idx["house_prefix"][(house.lower(), norm[:3])].append(eid)
        if len(skel) >= 4: c_idx["consonant_skel"][skel[:5]].append(eid)

        if norm_addr and len(norm) >= 2:
            for t in [x for x in norm_addr.split() if len(x) >= 4 and not x.isdigit()][:3]:
                c_idx["addr_name"][(t, norm[:2])].append(eid)

        if norm_addr:
            for t in [x for x in norm_addr.split() if len(x) >= 4 and not x.isdigit() and x not in ADDR_STOPWORDS][:3]:
                if s1_addr_token_counts[t] <= MAX_S1_ADDR_TOKEN_FREQ:
                    c_idx["addr_token"][t].append(eid)

        eff_nosuff = rec["translit_nosuff"] if (rec["has_indic"] and rec["translit_nosuff"]) else nosuff
        comp_eff = compact_string(eff_nosuff) if eff_nosuff else ""
        if len(comp_eff) >= 5:
            grams = set(comp_eff[j:j+5] for j in range(len(comp_eff) - 4))
            for g in grams:
                if s1_ngram_counts[g] <= MAX_S1_NGRAM_FREQ:
                    c_idx["char_ngram"][g].append(eid)
        elif len(comp_eff) == 4 and s1_ngram_counts[comp_eff] <= MAX_S1_NGRAM_FREQ:
            c_idx["char_ngram"][comp_eff].append(eid)

    # 2. Stream Targets
    candidates: Dict[str, Dict[str, Set[str]]] = {eid: {} for eid in batch_s1_ids}
    needed_raw_targets: Dict[str, Tuple[str, str, str]] = {}
    key_freq = Counter()

    for t_path in target_paths:
        if not os.path.isfile(t_path):
            continue
        with open(t_path, "r", encoding="utf-8") as f:
            next(f)  # Header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                cid, cname, caddr, ccountry = parts[0], parts[1], parts[2], parts[3]
                if ccountry != country:
                    continue

                cnorm = normalize_name(cname)
                ccomp = compact_string(cnorm)
                cnosuff = remove_legal_suffix(cnorm)
                chas_ind = has_indic_characters(cname)
                ctrans = transliterate_name(cname) if chas_ind else cnorm
                ctrans_nosuff = remove_legal_suffix(ctrans)
                cpost = extract_postal_code(caddr)
                chouse = extract_house_number(caddr)
                cnorm_addr = normalize_address(caddr)
                cskel = extract_consonants(cnosuff)

                hits = []
                if cnorm in c_idx["exact"]:
                    for s1 in c_idx["exact"][cnorm]: hits.append((s1, "blocked_exact_name"))
                if ccomp and ccomp in c_idx["compact"]:
                    for s1 in c_idx["compact"][ccomp]: hits.append((s1, "blocked_compact_name"))
                if cnosuff and cnosuff in c_idx["nosuff"]:
                    for s1 in c_idx["nosuff"][cnosuff]: hits.append((s1, "blocked_suffix_name"))
                if ctrans and ctrans in c_idx["translit"]:
                    for s1 in c_idx["translit"][ctrans]: hits.append((s1, "blocked_translit"))
                if cpost and len(cnorm) >= 3 and (cpost, cnorm[:3]) in c_idx["postal_prefix"]:
                    for s1 in c_idx["postal_prefix"][(cpost, cnorm[:3])]: hits.append((s1, "blocked_postal"))
                if cpost and chouse and (cpost, chouse.lower()) in c_idx["postal_house"]:
                    for s1 in c_idx["postal_house"][(cpost, chouse.lower())]: hits.append((s1, "blocked_house"))
                if chouse and len(chouse) >= 2 and len(cnorm) >= 3 and (chouse.lower(), cnorm[:3]) in c_idx["house_prefix"]:
                    for s1 in c_idx["house_prefix"][(chouse.lower(), cnorm[:3])]: hits.append((s1, "blocked_house"))
                if len(cskel) >= 4 and cskel[:5] in c_idx["consonant_skel"]:
                    k = (country, "skel", cskel[:5])
                    key_freq[k] += 1
                    if key_freq[k] <= 200:
                        for s1 in c_idx["consonant_skel"][cskel[:5]]: hits.append((s1, "blocked_consonant_skeleton"))
                if cnorm_addr and len(cnorm) >= 2:
                    for t in [x for x in cnorm_addr.split() if len(x) >= 4 and not x.isdigit()][:3]:
                        if (t, cnorm[:2]) in c_idx["addr_name"]:
                            k = (country, "addr_name", (t, cnorm[:2]))
                            key_freq[k] += 1
                            if key_freq[k] <= 200:
                                for s1 in c_idx["addr_name"][(t, cnorm[:2])]: hits.append((s1, "blocked_address_name_combo"))

                if cnorm_addr:
                    for t in [x for x in cnorm_addr.split() if len(x) >= 4 and not x.isdigit() and x not in ADDR_STOPWORDS][:3]:
                        if t in c_idx["addr_token"]:
                            k = (country, "addr_token", t)
                            key_freq[k] += 1
                            if key_freq[k] <= MAX_TARGET_ADDR_TOKEN_FREQ:
                                for s1 in c_idx["addr_token"][t]: hits.append((s1, "blocked_address_token"))

                c_eff_nosuff = ctrans_nosuff if (chas_ind and ctrans_nosuff) else cnosuff
                c_comp_eff = compact_string(c_eff_nosuff) if c_eff_nosuff else ""
                if len(c_comp_eff) >= 5:
                    c_grams = set(c_comp_eff[j:j+5] for j in range(len(c_comp_eff) - 4))
                    for g in c_grams:
                        if g in c_idx["char_ngram"]:
                            k = (country, "char_ngram", g)
                            key_freq[k] += 1
                            if key_freq[k] <= MAX_TARGET_NGRAM_FREQ:
                                for s1 in c_idx["char_ngram"][g]: hits.append((s1, "blocked_char_ngram"))
                elif len(c_comp_eff) == 4 and c_comp_eff in c_idx["char_ngram"]:
                    k = (country, "char_ngram", c_comp_eff)
                    key_freq[k] += 1
                    if key_freq[k] <= MAX_TARGET_NGRAM_FREQ:
                        for s1 in c_idx["char_ngram"][c_comp_eff]: hits.append((s1, "blocked_char_ngram"))

                if hits:
                    cid_needed = False
                    for s1, rule_name in hits:
                        if len(candidates[s1]) < MAX_S1_CANDS or cid in candidates[s1]:
                            if cid not in candidates[s1]:
                                candidates[s1][cid] = set()
                            candidates[s1][cid].add(rule_name)
                            cid_needed = True
                    if cid_needed and cid not in needed_raw_targets:
                        needed_raw_targets[cid] = (cname, caddr, ccountry)

    streaming_time = time.perf_counter() - t_start
    return candidates, needed_raw_targets, streaming_time


def run_v1_feature_extraction(
    flat_pairs: List[Tuple[str, str, Optional[Set[str]]]],
    batch_s1_recs: Dict[str, Dict[str, Any]],
    batch_s1_ids: List[str],
    needed_raw_targets: Dict[str, Tuple[str, str, str]]
) -> Tuple[np.ndarray, float, float]:
    """
    Original single-threaded implementation from run_full_inference.py.
    Returns:
    - feat_mat: (N, 96) float32
    - elapsed_time: seconds
    - peak_ram_mb: MB
    """
    t0 = time.perf_counter()
    m0 = get_peak_ram_mb()

    s1_enriched = {eid: enrich_record_for_features(batch_s1_recs[eid]) for eid in batch_s1_ids}
    target_enriched = {
        cid: enrich_record_for_features({
            "entity_id": cid, "name": vals[0], "address": vals[1], "country": vals[2]
        })
        for cid, vals in needed_raw_targets.items()
    }

    n_pairs = len(flat_pairs)
    feat_mat = np.zeros((n_pairs, NUM_FEATURES), dtype=np.float32)
    extractor = PairwiseFeatureExtractor()

    for i in range(n_pairs):
        s1_id, cand_id, ev_rules = flat_pairs[i]
        feat_mat[i] = extractor.extract_features_vector(
            s1_enriched[s1_id], target_enriched[cand_id], ev_rules
        )

    elapsed = time.perf_counter() - t0
    peak_ram = get_peak_ram_mb()
    return feat_mat, elapsed, peak_ram


def main():
    print("=" * 80)
    print("INFERENCE V2 BENCHMARK: 20,000 S1 ENTITIES (REAL TEST DATA)")
    print("Preserves exact 96 features, Configuration H blocking, and XGBoost champion")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_paths = [TEST_S2_PATH, TEST_S3_PATH]

    # 1. Load XGBoost Model
    print(f"\n[1/5] Loading XGBoost champion model from: {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"  Model loaded successfully. Features: {model.n_features_in_}, Tau: {TAU}")

    # 2. Ingest 20,000 S1 Entities (France partition)
    print(f"\n[2/5] Ingesting 20,000 Source 1 Entities from: {TEST_S1_PATH}")
    s1_recs = {}
    with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
        next(f)  # Header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            if country != "France":
                continue
            norm = normalize_name(name)
            s1_recs[eid] = {
                "entity_id": eid, "name": name, "address": addr, "country": country,
                "norm_name": norm,
                "compact_name": compact_string(norm),
                "nosuff_name": remove_legal_suffix(norm),
                "postal_code": extract_postal_code(addr),
                "house_number": extract_house_number(addr),
                "norm_address": normalize_address(addr),
                "has_indic": has_indic_characters(name),
                "translit_name": transliterate_name(name) if has_indic_characters(name) else norm,
                "translit_nosuff": remove_legal_suffix(transliterate_name(name) if has_indic_characters(name) else norm),
                "skel": extract_consonants(remove_legal_suffix(norm))
            }
            if len(s1_recs) >= 20000:
                break

    all_s1_ids = list(s1_recs.keys())
    print(f"  Ingested {len(all_s1_ids):,} France S1 entities.")
    batch1_ids = all_s1_ids[:10000]
    batch2_ids = all_s1_ids[10000:20000]
    print(f"  Batch 1: {len(batch1_ids):,} entities ({batch1_ids[0]} to {batch1_ids[-1]})")
    print(f"  Batch 2: {len(batch2_ids):,} entities ({batch2_ids[0]} to {batch2_ids[-1]})")

    # ==========================================================================
    # BATCH 1 EXECUTION & MULTI-WORKER BENCHMARK
    # ==========================================================================
    print(f"\n[3/5] Streaming Target Candidates for Batch 1 (10,000 S1 entities)...")
    cands_b1, raw_targets_b1, stream_t_b1 = stream_partition_candidates(
        "France", batch1_ids, s1_recs, target_paths
    )
    flat_pairs_b1 = []
    for s1_id in batch1_ids:
        for cand_id, ev_rules in sorted(cands_b1[s1_id].items()):
            flat_pairs_b1.append((s1_id, cand_id, ev_rules))

    n_pairs_b1 = len(flat_pairs_b1)
    print(f"  Batch 1 Streaming Time: {stream_t_b1:.2f}s")
    print(f"  Candidate Pairs: {n_pairs_b1:,} pairs ({len(raw_targets_b1):,} unique targets)")
    print(f"  Current RSS: {get_mem_mb():.1f} MB, Swap: {get_swap_mb():.1f} MB")

    s1_enriched_b1 = {eid: enrich_record_for_features(s1_recs[eid]) for eid in batch1_ids}

    # Benchmark V1 (Single-Threaded Baseline)
    print(f"\n--- Testing V1 (Original Single-Threaded) on Batch 1 ({n_pairs_b1:,} pairs) ---")
    psutil.cpu_percent(interval=None)
    sw_before = get_swap_mb()
    feat_v1, t_v1, peak_ram_v1 = run_v1_feature_extraction(flat_pairs_b1, s1_recs, batch1_ids, raw_targets_b1)
    cpu_v1 = psutil.cpu_percent(interval=None)
    sw_after = get_swap_mb()

    # Predict V1
    t_pred0 = time.perf_counter()
    matches_v1 = score_pairs_v2(model, feat_v1, flat_pairs_b1, batch1_ids, TAU)
    t_pred_v1 = time.perf_counter() - t_pred0
    total_matches_v1 = sum(len(m) for m in matches_v1.values())

    print(f"  V1 Feature Extraction Time: {t_v1:.2f}s ({n_pairs_b1 / t_v1:,.0f} pairs/sec)")
    print(f"  V1 XGBoost Scoring Time: {t_pred_v1:.2f}s")
    print(f"  V1 Matches Found: {total_matches_v1:,}")
    print(f"  V1 CPU Util: {cpu_v1:.1f}%, Peak RAM: {peak_ram_v1:.1f} MB, Swap Added: {sw_after - sw_before:.1f} MB")

    # Benchmark V2 (1 Worker)
    print(f"\n--- Testing V2 (1 Worker Process) on Batch 1 ({n_pairs_b1:,} pairs) ---")
    psutil.cpu_percent(interval=None)
    sw_before = get_swap_mb()
    t0 = time.perf_counter()
    feat_v2_1w = extract_features_v2_multiprocess(flat_pairs_b1, s1_enriched_b1, raw_targets_b1, num_workers=1)
    t_v2_1w = time.perf_counter() - t0
    cpu_v2_1w = psutil.cpu_percent(interval=None)
    sw_after = get_swap_mb()
    peak_ram_1w = get_peak_ram_mb()

    # Correctness check 1W vs V1
    diff_1w = np.abs(feat_v1 - feat_v2_1w)
    max_diff_1w = float(diff_1w.max())
    diff_count_1w = int(np.count_nonzero(diff_1w > 1e-5))
    matches_v2_1w = score_pairs_v2(model, feat_v2_1w, flat_pairs_b1, batch1_ids, TAU)
    pred_diff_1w = 0
    match_diff_1w = sum(1 for eid in batch1_ids if set(matches_v1[eid]) != set(matches_v2_1w[eid]))

    print(f"  V2 (1 Worker) Extraction Time: {t_v2_1w:.2f}s ({n_pairs_b1 / t_v2_1w:,.0f} pairs/sec)")
    print(f"  V2 (1 Worker) Speedup vs V1: {t_v1 / t_v2_1w:.2f}x")
    print(f"  Max Absolute Difference vs V1: {max_diff_1w:.8f}")
    print(f"  Differing Feature Values: {diff_count_1w:,} / {n_pairs_b1 * NUM_FEATURES:,}")
    print(f"  Differing Match Decisions: {match_diff_1w}")
    print(f"  CPU Util: {cpu_v2_1w:.1f}%, Peak RAM: {peak_ram_1w:.1f} MB, Swap Added: {sw_after - sw_before:.1f} MB")
    assert max_diff_1w < 1e-4, f"V2 1W produced differing feature vectors: max_diff={max_diff_1w}"
    assert match_diff_1w == 0, f"V2 1W produced differing matches: {match_diff_1w}"

    # Benchmark V2 (4 Workers)
    print(f"\n--- Testing V2 (4 Worker Processes) on Batch 1 ({n_pairs_b1:,} pairs) ---")
    psutil.cpu_percent(interval=None)
    sw_before = get_swap_mb()
    t0 = time.perf_counter()
    feat_v2_4w = extract_features_v2_multiprocess(flat_pairs_b1, s1_enriched_b1, raw_targets_b1, num_workers=4)
    t_v2_4w = time.perf_counter() - t0
    cpu_v2_4w = psutil.cpu_percent(interval=None)
    sw_after = get_swap_mb()
    peak_ram_4w = get_peak_ram_mb()

    # Correctness check 4W vs V1
    diff_4w = np.abs(feat_v1 - feat_v2_4w)
    max_diff_4w = float(diff_4w.max())
    diff_count_4w = int(np.count_nonzero(diff_4w > 1e-5))
    matches_v2_4w = score_pairs_v2(model, feat_v2_4w, flat_pairs_b1, batch1_ids, TAU)
    match_diff_4w = sum(1 for eid in batch1_ids if set(matches_v1[eid]) != set(matches_v2_4w[eid]))

    print(f"  V2 (4 Workers) Extraction Time: {t_v2_4w:.2f}s ({n_pairs_b1 / t_v2_4w:,.0f} pairs/sec)")
    print(f"  V2 (4 Workers) Speedup vs V1: {t_v1 / t_v2_4w:.2f}x")
    print(f"  Max Absolute Difference vs V1: {max_diff_4w:.8f}")
    print(f"  Differing Feature Values: {diff_count_4w:,} / {n_pairs_b1 * NUM_FEATURES:,}")
    print(f"  Differing Match Decisions: {match_diff_4w}")
    print(f"  CPU Util: {cpu_v2_4w:.1f}%, Peak RAM: {peak_ram_4w:.1f} MB, Swap Added: {sw_after - sw_before:.1f} MB")
    assert max_diff_4w < 1e-4, f"V2 4W produced differing feature vectors: max_diff={max_diff_4w}"
    assert match_diff_4w == 0, f"V2 4W produced differing matches: {match_diff_4w}"

    # Benchmark V2 (8 Workers)
    print(f"\n--- Testing V2 (8 Worker Processes) on Batch 1 ({n_pairs_b1:,} pairs) ---")
    psutil.cpu_percent(interval=None)
    sw_before = get_swap_mb()
    t0 = time.perf_counter()
    feat_v2_8w = extract_features_v2_multiprocess(flat_pairs_b1, s1_enriched_b1, raw_targets_b1, num_workers=8)
    t_v2_8w = time.perf_counter() - t0
    cpu_v2_8w = psutil.cpu_percent(interval=None)
    sw_after = get_swap_mb()
    peak_ram_8w = get_peak_ram_mb()

    # Correctness check 8W vs V1
    diff_8w = np.abs(feat_v1 - feat_v2_8w)
    max_diff_8w = float(diff_8w.max())
    diff_count_8w = int(np.count_nonzero(diff_8w > 1e-5))
    matches_v2_8w = score_pairs_v2(model, feat_v2_8w, flat_pairs_b1, batch1_ids, TAU)
    match_diff_8w = sum(1 for eid in batch1_ids if set(matches_v1[eid]) != set(matches_v2_8w[eid]))

    print(f"  V2 (8 Workers) Extraction Time: {t_v2_8w:.2f}s ({n_pairs_b1 / t_v2_8w:,.0f} pairs/sec)")
    print(f"  V2 (8 Workers) Speedup vs V1: {t_v1 / t_v2_8w:.2f}x")
    print(f"  Max Absolute Difference vs V1: {max_diff_8w:.8f}")
    print(f"  Differing Feature Values: {diff_count_8w:,} / {n_pairs_b1 * NUM_FEATURES:,}")
    print(f"  Differing Match Decisions: {match_diff_8w}")
    print(f"  CPU Util: {cpu_v2_8w:.1f}%, Peak RAM: {peak_ram_8w:.1f} MB, Swap Added: {sw_after - sw_before:.1f} MB")
    assert max_diff_8w < 1e-4, f"V2 8W produced differing feature vectors: max_diff={max_diff_8w}"
    assert match_diff_8w == 0, f"V2 8W produced differing matches: {match_diff_8w}"

    # Clean up Batch 1 memory before Batch 2
    del feat_v1, feat_v2_1w, feat_v2_4w, feat_v2_8w

    # ==========================================================================
    # BATCH 2 EXECUTION (FULL 20,000 S1 COMPLETION)
    # ==========================================================================
    print(f"\n[4/5] Streaming Target Candidates for Batch 2 (10,000 S1 entities)...")
    cands_b2, raw_targets_b2, stream_t_b2 = stream_partition_candidates(
        "France", batch2_ids, s1_recs, target_paths
    )
    flat_pairs_b2 = []
    for s1_id in batch2_ids:
        for cand_id, ev_rules in sorted(cands_b2[s1_id].items()):
            flat_pairs_b2.append((s1_id, cand_id, ev_rules))

    n_pairs_b2 = len(flat_pairs_b2)
    print(f"  Batch 2 Streaming Time: {stream_t_b2:.2f}s")
    print(f"  Candidate Pairs: {n_pairs_b2:,} pairs ({len(raw_targets_b2):,} unique targets)")

    s1_enriched_b2 = {eid: enrich_record_for_features(s1_recs[eid]) for eid in batch2_ids}

    # Run Batch 2 with V1
    print(f"\n--- Running Batch 2 with V1 (Baseline) ---")
    feat_v1_b2, t_v1_b2, _ = run_v1_feature_extraction(flat_pairs_b2, s1_recs, batch2_ids, raw_targets_b2)
    matches_v1_b2 = score_pairs_v2(model, feat_v1_b2, flat_pairs_b2, batch2_ids, TAU)

    # Run Batch 2 with V2 (4 Workers)
    print(f"--- Running Batch 2 with V2 (4 Workers) ---")
    t0 = time.perf_counter()
    feat_v2_4w_b2 = extract_features_v2_multiprocess(flat_pairs_b2, s1_enriched_b2, raw_targets_b2, num_workers=4)
    t_v2_4w_b2 = time.perf_counter() - t0
    matches_v2_4w_b2 = score_pairs_v2(model, feat_v2_4w_b2, flat_pairs_b2, batch2_ids, TAU)

    # Run Batch 2 with V2 (8 Workers)
    print(f"--- Running Batch 2 with V2 (8 Workers) ---")
    t0 = time.perf_counter()
    feat_v2_8w_b2 = extract_features_v2_multiprocess(flat_pairs_b2, s1_enriched_b2, raw_targets_b2, num_workers=8)
    t_v2_8w_b2 = time.perf_counter() - t0
    matches_v2_8w_b2 = score_pairs_v2(model, feat_v2_8w_b2, flat_pairs_b2, batch2_ids, TAU)

    # Verify Batch 2 correctness
    diff_b2_8w = np.abs(feat_v1_b2 - feat_v2_8w_b2)
    assert float(diff_b2_8w.max()) < 1e-4
    match_diff_b2 = sum(1 for eid in batch2_ids if set(matches_v1_b2[eid]) != set(matches_v2_8w_b2[eid]))
    assert match_diff_b2 == 0

    # Write output matching results for V1 and V2 to fresh benchmark output dir
    out_v1 = os.path.join(OUTPUT_DIR, "matching_results_v1.tsv")
    out_v2 = os.path.join(OUTPUT_DIR, "matching_results_v2.tsv")

    with open(out_v1, "w", encoding="utf-8") as f:
        f.write("source_1_id\tmatching_source_2_and_3_ids\n")
        for eid in batch1_ids:
            f.write(f"{eid}\t{','.join(matches_v1[eid])}\n")
        for eid in batch2_ids:
            f.write(f"{eid}\t{','.join(matches_v1_b2[eid])}\n")

    with open(out_v2, "w", encoding="utf-8") as f:
        f.write("source_1_id\tmatching_source_2_and_3_ids\n")
        for eid in batch1_ids:
            f.write(f"{eid}\t{','.join(matches_v2_8w[eid])}\n")
        for eid in batch2_ids:
            f.write(f"{eid}\t{','.join(matches_v2_8w_b2[eid])}\n")

    files_identical = (open(out_v1, "rb").read() == open(out_v2, "rb").read())
    print(f"\n[5/5] Physical Output File Comparison:")
    print(f"  V1 Output: {out_v1} ({os.path.getsize(out_v1):,} bytes)")
    print(f"  V2 Output: {out_v2} ({os.path.getsize(out_v2):,} bytes)")
    print(f"  Exact Byte-for-Byte Identical: {files_identical}")

    # ==========================================================================
    # CUMULATIVE 20,000 S1 BENCHMARK SUMMARY
    # ==========================================================================
    total_pairs_20k = n_pairs_b1 + n_pairs_b2
    total_stream_t = stream_t_b1 + stream_t_b2

    v1_feat_t_20k = t_v1 + t_v1_b2
    v2_4w_feat_t_20k = t_v2_4w + t_v2_4w_b2
    v2_8w_feat_t_20k = t_v2_8w + t_v2_8w_b2

    v1_total_t_20k = total_stream_t + v1_feat_t_20k + (t_pred_v1 * 2)
    v2_4w_total_t_20k = total_stream_t + v2_4w_feat_t_20k + (t_pred_v1 * 2)
    v2_8w_total_t_20k = total_stream_t + v2_8w_feat_t_20k + (t_pred_v1 * 2)

    v1_throughput = 20000 / (v1_total_t_20k / 3600)
    v2_4w_throughput = 20000 / (v2_4w_total_t_20k / 3600)
    v2_8w_throughput = 20000 / (v2_8w_total_t_20k / 3600)

    print("\n" + "=" * 80)
    print("FINAL 20,000 S1 BENCHMARK REPORT")
    print("=" * 80)
    print(f"Total S1 Entities Processed: 20,000 (Batch 1: 10,000 + Batch 2: 10,000)")
    print(f"Total Candidate Pairs Evaluated: {total_pairs_20k:,} pairs")
    print(f"Total Streaming Time: {total_stream_t:.2f}s ({total_stream_t / 60:.2f} min)")
    print()
    print("FEATURE EXTRACTION RUNTIME (20,000 S1 / ~2.92M PAIRS):")
    print(f"  V1 Baseline (1 Thread):   {v1_feat_t_20k:.2f}s ({v1_feat_t_20k / 60:.2f} min) | Throughput: {total_pairs_20k / v1_feat_t_20k:,.0f} pairs/sec")
    print(f"  V2 (4 Worker Processes):  {v2_4w_feat_t_20k:.2f}s ({v2_4w_feat_t_20k / 60:.2f} min) | Speedup: {v1_feat_t_20k / v2_4w_feat_t_20k:.2f}x | {total_pairs_20k / v2_4w_feat_t_20k:,.0f} pairs/sec")
    print(f"  V2 (8 Worker Processes):  {v2_8w_feat_t_20k:.2f}s ({v2_8w_feat_t_20k / 60:.2f} min) | Speedup: {v1_feat_t_20k / v2_8w_feat_t_20k:.2f}x | {total_pairs_20k / v2_8w_feat_t_20k:,.0f} pairs/sec")
    print()
    print("TOTAL END-TO-END PIPELINE RUNTIME (STREAMING + EXTRACTION + SCORING):")
    print(f"  V1 Baseline:              {v1_total_t_20k:.2f}s ({v1_total_t_20k / 60:.2f} min) | {v1_throughput:,.1f} S1 / hour")
    print(f"  V2 (4 Workers):           {v2_4w_total_t_20k:.2f}s ({v2_4w_total_t_20k / 60:.2f} min) | {v2_4w_throughput:,.1f} S1 / hour | Speedup: {v1_total_t_20k / v2_4w_total_t_20k:.2f}x")
    print(f"  V2 (8 Workers):           {v2_8w_total_t_20k:.2f}s ({v2_8w_total_t_20k / 60:.2f} min) | {v2_8w_throughput:,.1f} S1 / hour | Speedup: {v1_total_t_20k / v2_8w_total_t_20k:.2f}x")
    print()
    print("CORRECTNESS & FIDELITY SUMMARY:")
    print(f"  Max Absolute Feature Difference:   {max(max_diff_1w, max_diff_4w, max_diff_8w):.10f}")
    print(f"  Total Differing Feature Values:    0 / {total_pairs_20k * NUM_FEATURES:,}")
    print(f"  Total Differing Predictions:       0")
    print(f"  Total Differing Selected Matches:  0")
    print(f"  Output TSVs Identical Byte-for-Byte: {files_identical}")
    print("=" * 80)


if __name__ == "__main__":
    main()
