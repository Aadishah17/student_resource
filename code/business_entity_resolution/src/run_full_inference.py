"""
ML Challenge 2026: Business Entity Resolution
Module: run_full_inference.py

Full-scale test inference engine across all 1,732,545 test Source 1 entities:
- Dynamic country-partitioned execution across all discovered countries
- High-recall, memory-efficient Configuration H blocking
- Vectorized 96 handcrafted feature extraction in bounded streaming chunks
- Champion 50k XGBoost model scoring at threshold tau = 0.72
- Atomic, streaming output formatting for matching_results.tsv and candidate_pairs.tsv
- Fault-tolerant checkpointing with atomic resume capability
- Fully parameterized CLI arguments for datasets, models, metadata, and outputs
- Automatic execution of utils/validate_submission.py upon completion
"""

import os
import sys
import time
import json
import csv
import argparse
import gc
import psutil
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
import xgboost as xgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", "..", ".."))
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

DEFAULT_MODEL_PATH = os.path.join(SRC_DIR, "..", "models", "best_model.json")
DEFAULT_METADATA_PATH = os.path.join(SRC_DIR, "..", "models", "model_metadata.json")

DEFAULT_TEST_S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
DEFAULT_TEST_S2_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source2.tsv")
DEFAULT_TEST_S3_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source3.tsv")

DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEFAULT_MATCHING_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "matching_results.tsv")
DEFAULT_CANDIDATE_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "candidate_pairs.tsv")

DEFAULT_BATCH_SIZE = 10000
DEFAULT_MAX_S2_CANDS = 60
DEFAULT_MAX_S3_CANDS = 90
MAX_S1_CANDS = DEFAULT_MAX_S2_CANDS + DEFAULT_MAX_S3_CANDS  # 150 total ceiling
CHUNK_FEATURE_SIZE = 50000
MAX_S1_ADDR_TOKEN_FREQ = 5
MAX_S1_NGRAM_FREQ = 2
MAX_TARGET_ADDR_TOKEN_FREQ = DEFAULT_MAX_ADDR_TOKEN_FREQ  # 80
MAX_TARGET_NGRAM_FREQ = 50


def get_mem_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def save_checkpoint(ckpt_path: str, state: Dict[str, Any]):
    """Atomically writes checkpoint state to prevent corruption on sudden termination."""
    tmp_path = ckpt_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, ckpt_path)


def load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """Loads checkpoint state if present."""
    if os.path.isfile(ckpt_path):
        try:
            with open(ckpt_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {ckpt_path}: {e}")
    return {"completed_batches": [], "written_s1_count": 0}


def process_country_partition(
    country: str,
    s1_records: Dict[str, Dict[str, Any]],
    target_paths: List[str],
    model: xgb.XGBClassifier,
    tau: float,
    extractor: PairwiseFeatureExtractor,
    out_matching_file,
    out_candidate_file,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_s2_cands: int = DEFAULT_MAX_S2_CANDS,
    max_s3_cands: int = DEFAULT_MAX_S3_CANDS,
    checkpoint_file: Optional[str] = None,
    checkpoint_state: Optional[Dict[str, Any]] = None,
    exit_after_batches: Optional[int] = None,
    num_workers: int = 8,
    legacy_shared_quota: bool = False
):
    """
    Processes all Source 1 entities for a given country against test target sources.
    Supports atomic batch checkpointing so completed batches are never recomputed.
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING PARTITION: {country} ({len(s1_records):,} Source 1 Entities)")
    print(f"{'='*80}", flush=True)

    t_part_start = time.time()
    s1_ids = sorted(list(s1_records.keys()))
    n_total_s1 = len(s1_ids)
    max_s1_cands = 150 if legacy_shared_quota else (max_s2_cands + max_s3_cands)

    # Incompatible batch size and quota guards on checkpoint_state
    if checkpoint_state:
        ckpt_bs = checkpoint_state.get("batch_size")
        if ckpt_bs is not None and ckpt_bs != batch_size:
            raise ValueError(
                f"Incompatible batch size in checkpoint_state: checkpoint was created with batch_size={ckpt_bs}, "
                f"but process_country_partition was called with batch_size={batch_size}."
            )
        if not legacy_shared_quota:
            ckpt_s2 = checkpoint_state.get("max_s2_cands")
            ckpt_s3 = checkpoint_state.get("max_s3_cands")
            if ckpt_s2 is not None and ckpt_s2 != max_s2_cands:
                raise ValueError(
                    f"Incompatible S2 quota in checkpoint_state: checkpoint was created with max_s2_cands={ckpt_s2}, "
                    f"but process_country_partition was called with max_s2_cands={max_s2_cands}."
                )
            if ckpt_s3 is not None and ckpt_s3 != max_s3_cands:
                raise ValueError(
                    f"Incompatible S3 quota in checkpoint_state: checkpoint was created with max_s3_cands={ckpt_s3}, "
                    f"but process_country_partition was called with max_s3_cands={max_s3_cands}."
                )

    completed_batches = set(checkpoint_state.get("completed_batches", [])) if checkpoint_state else set()

    for b_idx in range(0, n_total_s1, batch_size):
        b_end = min(b_idx + batch_size, n_total_s1)
        batch_key = f"{country}_batch_{b_idx}_{b_end}"

        if batch_key in completed_batches:
            print(f"\n--- Batch {b_idx//batch_size + 1} / {int(np.ceil(n_total_s1/batch_size))}: [{b_idx:,} to {b_end:,}] [CHECKPOINT: SKIPPED] ---", flush=True)
            continue

        batch_s1_ids = s1_ids[b_idx:b_end]
        batch_s1_recs = {eid: s1_records[eid] for eid in batch_s1_ids}
        print(f"\n--- Batch {b_idx//batch_size + 1} / {int(np.ceil(n_total_s1/batch_size))}: S1 [{b_idx:,} to {b_end:,}] ---", flush=True)

        # 1. Build S1 inverted index for batch
        t0 = time.time()

        # Precompute address token and character n-gram frequencies across this S1 batch
        s1_addr_counts = Counter()
        s1_ngram_counts = Counter()
        for raw in batch_s1_recs.values():
            n_addr = raw["norm_address"]
            if n_addr:
                toks = set(t for t in n_addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS)
                for t in toks:
                    s1_addr_counts[t] += 1

            eff_nosuff = raw["translit_nosuff"] if (raw["has_indic"] and raw["translit_nosuff"]) else raw["nosuff_name"]
            comp_eff = compact_string(eff_nosuff) if eff_nosuff else ""
            if len(comp_eff) >= 5:
                for g in set(comp_eff[j:j+5] for j in range(len(comp_eff) - 4)):
                    s1_ngram_counts[g] += 1
            elif len(comp_eff) == 4:
                s1_ngram_counts[comp_eff] += 1

        c_idx: Dict[str, Dict[Any, List[str]]] = defaultdict(lambda: defaultdict(list))
        for eid, raw in batch_s1_recs.items():
            norm = raw["norm_name"]
            comp = raw["compact_name"]
            nosuff = raw["nosuff_name"]
            post = raw["postal_code"]
            house = raw["house_number"]
            trans = raw["translit_name"]
            trans_nosuff = raw["translit_nosuff"]
            skel = raw["skel"]
            norm_addr = raw["norm_address"]
            has_ind = raw["has_indic"]

            if norm and len(norm) >= 3: c_idx["exact_norm"][norm].append(eid)
            if comp and len(comp) >= 3: c_idx["exact_compact"][comp].append(eid)
            if nosuff and len(nosuff) >= 3:
                c_idx["nosuff"][nosuff].append(eid)
                comp_ns = compact_string(nosuff)
                if comp_ns != comp: c_idx["nosuff_compact"][comp_ns].append(eid)
            if has_ind and trans:
                c_idx["translit"][trans].append(eid)
                if trans_nosuff: c_idx["translit_nosuff"][trans_nosuff].append(eid)
            if post and len(norm) >= 3: c_idx["postal_prefix"][(post, norm[:3])].append(eid)
            tokens = norm.split()
            if post and tokens and len(tokens[0]) >= 3: c_idx["postal_token"][(post, tokens[0])].append(eid)
            if post and house: c_idx["postal_house"][(post, house.lower())].append(eid)
            if house and len(house) >= 2 and len(norm) >= 3: c_idx["house_prefix"][(house.lower(), norm[:3])].append(eid)
            if len(skel) >= 4: c_idx["consonant_skel"][skel[:5]].append(eid)
            if norm_addr and len(norm) >= 2:
                addr_tokens = [t for t in norm_addr.split() if len(t) >= 4 and not t.isdigit()]
                for t in addr_tokens[:3]: c_idx["addr_name"][(t, norm[:2])].append(eid)

            # Rare discriminating address tokens (blocked_address_token)
            if norm_addr:
                rare_addr_toks = [t for t in norm_addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and s1_addr_counts[t] <= MAX_S1_ADDR_TOKEN_FREQ]
                for t in rare_addr_toks[:3]:
                    c_idx["addr_token"][t].append(eid)

            # Rare character n-grams for fuzzy typos / variation (blocked_char_ngram)
            eff_nosuff = trans_nosuff if (has_ind and trans_nosuff) else nosuff
            comp_eff = compact_string(eff_nosuff) if eff_nosuff else ""
            if len(comp_eff) >= 5:
                rare_grams = [comp_eff[j:j+5] for j in range(len(comp_eff) - 4) if s1_ngram_counts[comp_eff[j:j+5]] <= MAX_S1_NGRAM_FREQ]
                for g in rare_grams:
                    c_idx["char_ngram"][g].append(eid)
            elif len(comp_eff) == 4 and s1_ngram_counts[comp_eff] <= MAX_S1_NGRAM_FREQ:
                c_idx["char_ngram"][comp_eff].append(eid)

        print(f"  Batch index built in {time.time()-t0:.2f}s, RAM: {get_mem_mb():.2f} MB")

        # 2. Stream target records for this country
        t_cand = time.time()
        candidates: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        s2_counts: Counter = Counter()
        s3_counts: Counter = Counter()
        needed_targets: Dict[str, Dict[str, Any]] = {}
        key_freq = Counter()

        for fpath in target_paths:
            if not os.path.isfile(fpath):
                print(f"  Warning: Target file not found: {fpath}")
                continue

            is_s2_file = "source2" in fpath.lower()

            with open(fpath, "r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r)
                for row in r:
                    if row[3] != country:
                        continue
                    cid, cname, caddr, ccountry = row[0], row[1], row[2], row[3]
                    is_s2 = cid.startswith("S2-") or is_s2_file
                    max_source_quota = max_s2_cands if is_s2 else max_s3_cands
                    source_counts = s2_counts if is_s2 else s3_counts

                    cnorm = normalize_name(cname)
                    ccomp = compact_string(cnorm)
                    cnosuff = remove_legal_suffix(cnorm)
                    cpost = extract_postal_code(caddr)
                    chouse = extract_house_number(caddr)
                    chas_ind = has_indic_characters(cname)
                    ctrans = transliterate_name(cname) if chas_ind else cnorm
                    ctrans_nosuff = remove_legal_suffix(ctrans)
                    cskel = extract_consonants(cnosuff)
                    cnorm_addr = normalize_address(caddr)

                    hits = []
                    if cnorm in c_idx["exact_norm"]:
                        for s1 in c_idx["exact_norm"][cnorm]: hits.append((s1, "blocked_exact_name"))
                    if ccomp in c_idx["exact_compact"]:
                        for s1 in c_idx["exact_compact"][ccomp]: hits.append((s1, "blocked_compact_name"))
                    if cnosuff in c_idx["nosuff"]:
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

                    # Rare discriminating address tokens (blocked_address_token)
                    if cnorm_addr:
                        for t in [x for x in cnorm_addr.split() if len(x) >= 4 and not x.isdigit() and x not in ADDR_STOPWORDS][:3]:
                            if t in c_idx["addr_token"]:
                                k = (country, "addr_token", t)
                                key_freq[k] += 1
                                if key_freq[k] <= MAX_TARGET_ADDR_TOKEN_FREQ:
                                    for s1 in c_idx["addr_token"][t]: hits.append((s1, "blocked_address_token"))

                    # Character n-gram fuzzy matching (blocked_char_ngram)
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
                        if legacy_shared_quota:
                            for s1, rule_name in hits:
                                if cid in candidates[s1]:
                                    candidates[s1][cid].add(rule_name)
                                    cid_needed = True
                                elif len(candidates[s1]) < max_s1_cands:
                                    candidates[s1][cid].add(rule_name)
                                    cid_needed = True
                        else:
                            for s1, rule_name in hits:
                                if cid in candidates[s1]:
                                    # Target already admitted for this S1: merge rule without consuming additional quota slot
                                    candidates[s1][cid].add(rule_name)
                                    cid_needed = True
                                elif source_counts[s1] < max_source_quota and len(candidates[s1]) < max_s1_cands:
                                    # New candidate within source-specific quota and total ceiling
                                    candidates[s1][cid].add(rule_name)
                                    source_counts[s1] += 1
                                    cid_needed = True

                        if cid_needed and cid not in needed_targets:
                            needed_targets[cid] = {
                                "entity_id": cid, "name": cname, "address": caddr, "country": ccountry
                            }

        n_cands_batch = sum(len(c) for c in candidates.values())
        print(f"  Streaming complete in {time.time()-t_cand:.2f}s: {n_cands_batch:,} pairs ({len(needed_targets):,} unique targets), RAM: {get_mem_mb():.2f} MB")

        # Explicitly free indexing structures to maximize RAM headroom before multiprocessing
        del c_idx, key_freq, s1_addr_counts, s1_ngram_counts
        gc.collect()

        # 3. Feature Extraction & Inference
        t_inf = time.time()
        s1_enriched = {eid: enrich_record_for_features(batch_s1_recs[eid]) for eid in batch_s1_ids}

        flat_pairs = []
        for s1_id in batch_s1_ids:
            for cand_id, ev_rules in sorted(candidates[s1_id].items()):
                flat_pairs.append((s1_id, cand_id, ev_rules))

        n_pairs = len(flat_pairs)
        matches_by_s1: Dict[str, List[str]] = {eid: [] for eid in batch_s1_ids}

        if n_pairs > 0:
            if num_workers > 1:
                raw_targets_map = {
                    cid: (rec["name"], rec["address"], rec["country"])
                    for cid, rec in needed_targets.items()
                }
                feat_mat = extract_features_v2_multiprocess(
                    flat_pairs, s1_enriched, raw_targets_map, num_workers=num_workers
                )
                matches_by_s1 = score_pairs_v2(
                    model, feat_mat, flat_pairs, batch_s1_ids, tau=tau
                )
                del feat_mat
            else:
                target_enriched = {cid: enrich_record_for_features(rec) for cid, rec in needed_targets.items()}
                for c_start in range(0, n_pairs, CHUNK_FEATURE_SIZE):
                    c_end = min(c_start + CHUNK_FEATURE_SIZE, n_pairs)
                    chunk_len = c_end - c_start
                    feat_mat = np.zeros((chunk_len, NUM_FEATURES), dtype=np.float32)

                    for i in range(chunk_len):
                        s1_id, cand_id, ev_rules = flat_pairs[c_start + i]
                        feat_mat[i] = extractor.extract_features_vector(s1_enriched[s1_id], target_enriched[cand_id], ev_rules)

                    probs = model.predict_proba(feat_mat)[:, 1]
                    for i in range(chunk_len):
                        if probs[i] >= tau:
                            s1_id, cand_id, _ = flat_pairs[c_start + i]
                            matches_by_s1[s1_id].append(cand_id)
                del target_enriched

        print(f"  Scoring complete in {time.time()-t_inf:.2f}s, RAM: {get_mem_mb():.2f} MB")

        # 4. Stream results to output files
        for eid in batch_s1_ids:
            matches_str = ",".join(matches_by_s1[eid])
            out_matching_file.write(f"{eid}\t{matches_str}\n")

            cands_str = ",".join(sorted(list(candidates[eid].keys())))
            out_candidate_file.write(f"{eid}\t{cands_str}\n")

        out_matching_file.flush()
        out_candidate_file.flush()

            # 5. Atomically update checkpoint
        if checkpoint_file and checkpoint_state is not None:
            cand_lens = [len(candidates[eid]) for eid in batch_s1_ids]
            b_mean_cands = float(np.mean(cand_lens)) if cand_lens else 0.0
            b_p95_cands = float(np.percentile(cand_lens, 95)) if cand_lens else 0.0
            b_s2_cands = sum(s2_counts[eid] for eid in batch_s1_ids)
            b_s3_cands = sum(s3_counts[eid] for eid in batch_s1_ids)
            b_total_cands = b_s2_cands + b_s3_cands
            pct_s2 = (b_s2_cands / b_total_cands * 100) if b_total_cands > 0 else 0.0
            pct_s3 = (b_s3_cands / b_total_cands * 100) if b_total_cands > 0 else 0.0
            b_elapsed = time.time() - t0
            throughput = len(batch_s1_ids) / b_elapsed if b_elapsed > 0 else 0.0
            peak_ram = get_mem_mb()

            completed_batches.add(batch_key)
            checkpoint_state["completed_batches"] = sorted(list(completed_batches))
            checkpoint_state["written_s1_count"] = checkpoint_state.get("written_s1_count", 0) + len(batch_s1_ids)
            checkpoint_state["batch_size"] = batch_size
            if not legacy_shared_quota:
                checkpoint_state["max_s2_cands"] = max_s2_cands
                checkpoint_state["max_s3_cands"] = max_s3_cands
            checkpoint_state["max_s1_cands"] = max_s1_cands
            checkpoint_state["matching_byte_offset"] = out_matching_file.tell()
            checkpoint_state["candidate_byte_offset"] = out_candidate_file.tell()
            checkpoint_state["last_country"] = country
            checkpoint_state["last_batch"] = batch_key
            checkpoint_state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            checkpoint_state["last_metrics"] = {
                "batch_elapsed_s": round(b_elapsed, 2),
                "throughput_s1_per_s": round(throughput, 2),
                "pct_s2_cands": round(pct_s2, 2),
                "pct_s3_cands": round(pct_s3, 2),
                "mean_cands_per_s1": round(b_mean_cands, 2),
                "p95_cands_per_s1": round(b_p95_cands, 2),
                "peak_ram_mb": round(peak_ram, 2)
            }
            save_checkpoint(checkpoint_file, checkpoint_state)
            print(
                f"  [CHECKPOINT] Saved: {batch_key} | S1 committed: {checkpoint_state['written_s1_count']:,} | "
                f"Batch time: {b_elapsed:.1f}s ({throughput:.1f} S1/s) | "
                f"Share: {pct_s2:.1f}% S2 / {pct_s3:.1f}% S3 | "
                f"Cands/S1: mean={b_mean_cands:.1f}, P95={b_p95_cands:.1f} | "
                f"RAM: {peak_ram:.1f} MB",
                flush=True
            )

            if exit_after_batches is not None and len(checkpoint_state["completed_batches"]) >= exit_after_batches:
                print(f"\n[SIMULATED INTERRUPTION] Reached limit of {exit_after_batches} completed batches. Exiting for resume testing.")
                sys.exit(0)

    print(f"Partition {country} finished in {time.time() - t_part_start:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Full AWS Test Inference Pipeline")
    parser.add_argument("--test-s1-path", type=str, default=DEFAULT_TEST_S1_PATH, help="Path to test_source1.tsv")
    parser.add_argument("--test-s2-path", type=str, default=DEFAULT_TEST_S2_PATH, help="Path to test_source2.tsv")
    parser.add_argument("--test-s3-path", type=str, default=DEFAULT_TEST_S3_PATH, help="Path to test_source3.tsv")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to champion model (best_model.json)")
    parser.add_argument("--metadata-path", type=str, default=DEFAULT_METADATA_PATH, help="Path to model_metadata.json")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save output files")
    parser.add_argument("--matching-out", type=str, default=None, help="Explicit path for matching_results.tsv (overrides output-dir)")
    parser.add_argument("--candidate-out", type=str, default=None, help="Explicit path for candidate_pairs.tsv (overrides output-dir)")
    parser.add_argument("--threshold", type=float, default=None, help="Classification probability threshold (default: read from metadata)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="S1 batch size per country iteration")
    parser.add_argument("--max-s2-cands", type=int, default=DEFAULT_MAX_S2_CANDS, help=f"Max candidates per S1 from Source 2 (default: {DEFAULT_MAX_S2_CANDS})")
    parser.add_argument("--max-s3-cands", type=int, default=DEFAULT_MAX_S3_CANDS, help=f"Max candidates per S1 from Source 3 (default: {DEFAULT_MAX_S3_CANDS})")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint if available")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume and restart fresh")
    parser.add_argument("--force-fresh", action="store_true", default=False, help="Force fresh start and overwrite outputs")
    parser.add_argument("--max-s1", type=int, default=None, help="Maximum number of S1 entities to process (for smoke testing)")
    parser.add_argument("--exit-after-batches", type=int, default=None, help="Simulate process interruption by exiting after N completed batches")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Explicit path to checkpoint file (default: .inference_checkpoint.json in output-dir)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes for V2 feature extraction (default: 8)")
    parser.add_argument("--legacy-shared-quota", action="store_true", default=False, help="Use legacy shared-150 candidate pool (for resuming legacy production runs)")
    parser.add_argument("--validate", action="store_true", default=True, help="Run validate_submission.py upon completion")
    args = parser.parse_args()

    max_s1_cands = 150 if args.legacy_shared_quota else (args.max_s2_cands + args.max_s3_cands)

    t_master = time.time()
    print("=" * 80)
    print("ML CHALLENGE 2026: FULL-SCALE TEST INFERENCE PIPELINE")
    print("=" * 80)

    # 1. Resolve Output Paths & Checkpoint
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    matching_out = os.path.abspath(args.matching_out) if args.matching_out else os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.abspath(args.candidate_out) if args.candidate_out else os.path.join(output_dir, "candidate_pairs.tsv")
    checkpoint_file = os.path.abspath(args.checkpoint_file) if args.checkpoint_file else os.path.join(output_dir, ".inference_checkpoint.json")

    print(f"Paths Configuration:")
    print(f"  - S1 Dataset   : {args.test_s1_path}")
    print(f"  - S2 Dataset   : {args.test_s2_path}")
    print(f"  - S3 Dataset   : {args.test_s3_path}")
    print(f"  - Model Path   : {args.model_path}")
    print(f"  - Metadata     : {args.metadata_path}")
    print(f"  - Matching Out : {matching_out}")
    print(f"  - Candidate Out: {candidate_out}")
    print(f"  - Checkpoint   : {checkpoint_file}")
    print(f"  - S2 Max Quota : {args.max_s2_cands}")
    print(f"  - S3 Max Quota : {args.max_s3_cands}")
    print(f"  - Total Max Cap: {max_s1_cands}")

    # 2. Load Model & Finalized Metadata
    print("\n[1/4] Loading Champion Model & Finalized Metadata...")
    assert os.path.isfile(args.metadata_path), f"Metadata not found at {args.metadata_path}"
    assert os.path.isfile(args.model_path), f"Model not found at {args.model_path}"

    with open(args.metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tau = args.threshold if args.threshold is not None else meta.get("selected_threshold", 0.72)
    expected_features = meta.get("feature_names", [])

    print(f"  Architecture   : {meta.get('model_architecture', 'XGBoost Champion')}")
    print(f"  Threshold      : {tau:.2f} (from {'CLI' if args.threshold is not None else 'metadata'})")
    print(f"  Feature Schema : {len(expected_features)} features (Config H + 96 Handcrafted)")

    assert len(expected_features) == 96, f"Expected 96 features, found {len(expected_features)}"

    model = xgb.XGBClassifier()
    model.load_model(args.model_path)
    extractor = PairwiseFeatureExtractor()
    assert extractor.feature_names == expected_features, "Feature order mismatch with metadata!"

    # 3. Ingest All Source 1 Entities & Dynamically Discover Countries
    print("\n[2/4] Ingesting Source 1 Entities & Discovering Countries Dynamically...")
    assert os.path.isfile(args.test_s1_path), f"Source 1 file not found at {args.test_s1_path}"

    s1_by_country: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    s1_count = 0

    with open(args.test_s1_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        for row in r:
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            norm = normalize_name(name)
            s1_by_country[country][eid] = {
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
            s1_count += 1
            if args.max_s1 is not None and s1_count >= args.max_s1:
                break

    discovered_countries = sorted(list(s1_by_country.keys()))
    print(f"  Dynamically Discovered Countries ({len(discovered_countries)}): {discovered_countries}")
    print(f"  Total Ingested Source 1 Entities: {s1_count:,}")
    for ctry in discovered_countries:
        cnt = len(s1_by_country[ctry])
        print(f"    - {ctry:15s}: {cnt:,} entities ({cnt/s1_count*100:.1f}%)")

    # 4. Determine Resume vs Fresh Execution
    can_resume = (
        args.resume
        and not args.force_fresh
        and os.path.isfile(checkpoint_file)
        and os.path.isfile(matching_out)
        and os.path.isfile(candidate_out)
        and os.path.getsize(matching_out) > 0
    )

    if can_resume:
        checkpoint_state = load_checkpoint(checkpoint_file)
        completed_batches_count = len(checkpoint_state.get("completed_batches", []))
        ckpt_batch_size = checkpoint_state.get("batch_size")

        # Fallback inference of batch_size from completed_batches keys if not explicitly recorded
        if ckpt_batch_size is None and checkpoint_state.get("completed_batches"):
            for b_key in checkpoint_state["completed_batches"]:
                parts = b_key.rsplit("_", 2)
                if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
                    span = int(parts[-1]) - int(parts[-2])
                    if span > 0:
                        ckpt_batch_size = span
                        break

        if ckpt_batch_size is not None and ckpt_batch_size != args.batch_size:
            err_msg = (
                f"FATAL: Incompatible batch size for resume. "
                f"Checkpoint was created with batch_size={ckpt_batch_size}, "
                f"but current execution requested batch_size={args.batch_size}. "
                f"Resuming with a different batch size will corrupt partition boundaries and produce duplicate or skipped records. "
                f"To resume safely, pass '--batch-size {ckpt_batch_size}', or specify '--force-fresh' to restart."
            )
            print(f"\n[ERROR] {err_msg}", file=sys.stderr)
            raise ValueError(err_msg)

        ckpt_max_s2 = checkpoint_state.get("max_s2_cands")
        ckpt_max_s3 = checkpoint_state.get("max_s3_cands")

        if not args.legacy_shared_quota:
            if ckpt_max_s2 is None or ckpt_max_s3 is None:
                err_msg = (
                    f"FATAL: Incompatible candidate quota for resume. "
                    f"Checkpoint was created with legacy shared candidate pool (flawed S2-first quota), "
                    f"but current execution requires source-separated quotas ({args.max_s2_cands} S2 / {args.max_s3_cands} S3). "
                    f"Resuming would mix flawed shared-quota batches with balanced batches. "
                    f"To restart fresh with the new quota, specify '--force-fresh'. "
                    f"To resume this run using legacy shared quota, specify '--legacy-shared-quota'."
                )
                print(f"\n[ERROR] {err_msg}", file=sys.stderr)
                raise ValueError(err_msg)

            if ckpt_max_s2 != args.max_s2_cands or ckpt_max_s3 != args.max_s3_cands:
                err_msg = (
                    f"FATAL: Incompatible candidate quota for resume. "
                    f"Checkpoint was created with max_s2_cands={ckpt_max_s2}, max_s3_cands={ckpt_max_s3}, "
                    f"but current execution requested max_s2_cands={args.max_s2_cands}, max_s3_cands={args.max_s3_cands}. "
                    f"Resuming with different quotas will produce inconsistent candidate distributions. "
                    f"To restart fresh with the new quota, specify '--force-fresh'."
                )
                print(f"\n[ERROR] {err_msg}", file=sys.stderr)
                raise ValueError(err_msg)
        else:
            print(f"  [LEGACY SHARED-QUOTA RESUME ACTIVE]")
            print(f"  Resuming existing production run with shared MAX_S1_CANDS={max_s1_cands} candidate pool.")

        print(f"\n[RESUME MODE ACTIVE]")
        print(f"  Existing checkpoint loaded from: {checkpoint_file}")
        print(f"  Already completed batches: {completed_batches_count}")
        print(f"  Verified batch size: {args.batch_size}")
        if args.legacy_shared_quota:
            print(f"  Verified quota policy: Legacy Shared 150 pool")
        else:
            print(f"  Verified quotas: {args.max_s2_cands} S2 / {args.max_s3_cands} S3 (Total: {max_s1_cands})")

        # Roll back uncompleted partial writes if process was killed mid-batch
        m_offset = checkpoint_state.get("matching_byte_offset")
        c_offset = checkpoint_state.get("candidate_byte_offset")
        if m_offset is not None and os.path.isfile(matching_out) and os.path.getsize(matching_out) > m_offset:
            print(f"  Rolling back matching file to last clean checkpoint offset ({m_offset:,} bytes)")
            with open(matching_out, "r+", encoding="utf-8") as fm:
                fm.seek(m_offset)
                fm.truncate()
        if c_offset is not None and os.path.isfile(candidate_out) and os.path.getsize(candidate_out) > c_offset:
            print(f"  Rolling back candidate file to last clean checkpoint offset ({c_offset:,} bytes)")
            with open(candidate_out, "r+", encoding="utf-8") as fc:
                fc.seek(c_offset)
                fc.truncate()

        print(f"  Appending new results to existing output files (mode='a').")
        file_mode = "a"
        write_headers = False
    else:
        if args.force_fresh:
            print(f"\n[FRESH MODE ACTIVE: --force-fresh specified]")
        else:
            print(f"\n[FRESH MODE ACTIVE: No prior checkpoint found]")
        if os.path.isfile(checkpoint_file):
            os.remove(checkpoint_file)
        checkpoint_state = {
            "completed_batches": [],
            "written_s1_count": 0,
            "batch_size": args.batch_size,
            "max_s2_cands": args.max_s2_cands,
            "max_s3_cands": args.max_s3_cands,
            "max_s1_cands": max_s1_cands,
            "threshold": tau,
            "model_path": os.path.abspath(args.model_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
        file_mode = "w"
        write_headers = True

    # 5. Execute Country-Partitioned Inference
    print("\n[3/4] Executing Dynamic Country Partition Inference...")
    target_paths = [args.test_s2_path, args.test_s3_path]

    with open(matching_out, file_mode, encoding="utf-8", newline="") as f_match, \
         open(candidate_out, file_mode, encoding="utf-8", newline="") as f_cand:

        if write_headers:
            f_match.write("source1_entity_id\tmatched_entity_ids\n")
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match.flush()
            f_cand.flush()
            checkpoint_state["matching_byte_offset"] = f_match.tell()
            checkpoint_state["candidate_byte_offset"] = f_cand.tell()
            checkpoint_state["batch_size"] = args.batch_size
            if not args.legacy_shared_quota:
                checkpoint_state["max_s2_cands"] = args.max_s2_cands
                checkpoint_state["max_s3_cands"] = args.max_s3_cands
            checkpoint_state["max_s1_cands"] = max_s1_cands
            save_checkpoint(checkpoint_file, checkpoint_state)

        for country in discovered_countries:
            process_country_partition(
                country=country,
                s1_records=s1_by_country[country],
                target_paths=target_paths,
                model=model,
                tau=tau,
                extractor=extractor,
                out_matching_file=f_match,
                out_candidate_file=f_cand,
                batch_size=args.batch_size,
                max_s2_cands=args.max_s2_cands,
                max_s3_cands=args.max_s3_cands,
                checkpoint_file=checkpoint_file,
                checkpoint_state=checkpoint_state,
                exit_after_batches=args.exit_after_batches,
                num_workers=args.workers,
                legacy_shared_quota=args.legacy_shared_quota
            )

    print(f"\nInference completed successfully!")
    print(f"  - Matching TSV  : {matching_out} ({os.path.getsize(matching_out):,} bytes)")
    print(f"  - Candidate TSV : {candidate_out} ({os.path.getsize(candidate_out):,} bytes)")

    # 6. Official Submission Validator Execution
    validator_path = os.path.join(BASE_DIR, "utils", "validate_submission.py")
    test_dir = os.path.dirname(args.test_s1_path)
    if args.validate and os.path.isfile(validator_path):
        print("\n[4/4] Running Official validate_submission.py...")
        import subprocess
        cmd = [
            sys.executable,
            validator_path,
            "--matching", matching_out,
            "--candidate", candidate_out,
            "--test-dir", test_dir
        ]
        print(f"  Command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print("VALIDATION SUCCESS: Submission files are 100% compliant with challenge scoring rules!")
        else:
            print(f"VALIDATION WARNING: Validator returned exit code {res.returncode}")

    print(f"\nTotal Pipeline Execution Time: {time.time() - t_master:.2f}s ({(time.time() - t_master)/60:.2f} min)")


if __name__ == "__main__":
    main()
