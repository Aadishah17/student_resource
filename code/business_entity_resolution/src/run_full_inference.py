"""
ML Challenge 2026: Business Entity Resolution
Module: run_full_inference.py

Full-scale test inference engine across all 1,732,545 test Source 1 entities:
- Country-partitioned execution (France -> US -> India)
- High-recall, memory-efficient Configuration H blocking
- Vectorized 96 handcrafted feature extraction in streaming chunks
- Champion 50k XGBoost model scoring at threshold tau = 0.72
- Atomic, streaming output formatting for matching_results.tsv and candidate_pairs.tsv
- Automatic execution of utils/validate_submission.py upon completion
"""

import os
import sys
import time
import json
import csv
import argparse
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
from blocking import extract_consonants
from features import (
    PairwiseFeatureExtractor,
    enrich_record_for_features,
    FEATURE_NAMES,
    NUM_FEATURES
)

MODEL_PATH = os.path.join(SRC_DIR, "..", "models", "best_model.json")
METADATA_PATH = os.path.join(SRC_DIR, "..", "models", "model_metadata.json")

TEST_S1_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source1.tsv")
TEST_S2_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source2.tsv")
TEST_S3_PATH = os.path.join(BASE_DIR, "dataset", "test", "test_source3.tsv")

DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEFAULT_MATCHING_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "matching_results.tsv")
DEFAULT_CANDIDATE_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "candidate_pairs.tsv")

MAX_S1_CANDS = 150
CHUNK_FEATURE_SIZE = 50000


def get_mem_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def process_country_partition(
    country: str,
    s1_records: Dict[str, Dict[str, Any]],
    model: xgb.XGBClassifier,
    tau: float,
    extractor: PairwiseFeatureExtractor,
    out_matching_file,
    out_candidate_file,
    batch_size: int = 50000
):
    """
    Processes all Source 1 entities for a given country against test_source2 and test_source3.
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING PARTITION: {country} ({len(s1_records):,} Source 1 Entities)")
    print(f"{'='*80}", flush=True)

    t_part_start = time.time()
    s1_ids = sorted(list(s1_records.keys()))
    n_total_s1 = len(s1_ids)

    # Process S1 in manageable batches to maintain bounded memory footprint
    for b_idx in range(0, n_total_s1, batch_size):
        b_end = min(b_idx + batch_size, n_total_s1)
        batch_s1_ids = s1_ids[b_idx:b_end]
        batch_s1_recs = {eid: s1_records[eid] for eid in batch_s1_ids}
        print(f"\n--- Batch {b_idx//batch_size + 1} / {int(np.ceil(n_total_s1/batch_size))}: S1 [{b_idx:,} to {b_end:,}] ---", flush=True)

        # 1. Build S1 inverted index for batch
        t0 = time.time()
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

        print(f"  Batch index built in {time.time()-t0:.2f}s, RAM: {get_mem_mb():.2f} MB")

        # 2. Stream target records for this country
        t_cand = time.time()
        candidates: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        needed_targets: Dict[str, Dict[str, Any]] = {}
        key_freq = Counter()

        for fpath in [TEST_S2_PATH, TEST_S3_PATH]:
            with open(fpath, "r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r)
                for row in r:
                    if row[3] != country:
                        continue
                    cid, cname, caddr, ccountry = row[0], row[1], row[2], row[3]

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

                    if hits:
                        cid_needed = False
                        for s1, rule_name in hits:
                            if len(candidates[s1]) < MAX_S1_CANDS or cid in candidates[s1]:
                                candidates[s1][cid].add(rule_name)
                                cid_needed = True
                        if cid_needed and cid not in needed_targets:
                            needed_targets[cid] = {
                                "entity_id": cid, "name": cname, "address": caddr, "country": ccountry
                            }

        n_cands_batch = sum(len(c) for c in candidates.values())
        print(f"  Streaming complete in {time.time()-t_cand:.2f}s: {n_cands_batch:,} pairs ({len(needed_targets):,} unique targets), RAM: {get_mem_mb():.2f} MB")

        # 3. Feature Extraction & Inference
        t_inf = time.time()
        s1_enriched = {eid: enrich_record_for_features(batch_s1_recs[eid]) for eid in batch_s1_ids}
        target_enriched = {cid: enrich_record_for_features(rec) for cid, rec in needed_targets.items()}

        flat_pairs = []
        for s1_id in batch_s1_ids:
            for cand_id, ev_rules in sorted(candidates[s1_id].items()):
                flat_pairs.append((s1_id, cand_id, ev_rules))

        n_pairs = len(flat_pairs)
        matches_by_s1: Dict[str, List[str]] = {eid: [] for eid in batch_s1_ids}

        if n_pairs > 0:
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

        print(f"  Scoring complete in {time.time()-t_inf:.2f}s, RAM: {get_mem_mb():.2f} MB")

        # 4. Stream results to output files
        for eid in batch_s1_ids:
            matches_str = ",".join(matches_by_s1[eid])
            out_matching_file.write(f"{eid}\t{matches_str}\n")

            cands_str = ",".join(sorted(list(candidates[eid].keys())))
            out_candidate_file.write(f"{eid}\t{cands_str}\n")

        out_matching_file.flush()
        out_candidate_file.flush()

    print(f"Partition {country} finished in {time.time() - t_part_start:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Full AWS Test Inference Pipeline")
    parser.add_argument("--batch-size", type=int, default=50000, help="S1 batch size per iteration")
    parser.add_argument("--matching-out", type=str, default=DEFAULT_MATCHING_PATH, help="Path for matching_results.tsv")
    parser.add_argument("--candidate-out", type=str, default=DEFAULT_CANDIDATE_PATH, help="Path for candidate_pairs.tsv")
    parser.add_argument("--validate", action="store_true", default=True, help="Run validate_submission.py upon completion")
    args = parser.parse_args()

    t_master = time.time()
    print("=" * 80)
    print("ML CHALLENGE 2026: FULL-SCALE TEST INFERENCE PIPELINE")
    print("=" * 80)

    # 1. Load Model & Metadata
    print("\n[1/4] Loading Champion Model & Finalized Metadata...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tau = meta["selected_threshold"]
    expected_features = meta["feature_names"]
    print(f"  Architecture : {meta['model_architecture']}")
    print(f"  Threshold    : {tau:.2f}")
    print(f"  Features     : {len(expected_features)}")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    extractor = PairwiseFeatureExtractor()
    assert extractor.feature_names == expected_features, "Feature order mismatch!"

    # 2. Ingest & Partition All Source 1 Entities by Country
    print("\n[2/4] Ingesting all Source 1 Test Entities from test_source1.tsv...")
    s1_by_country: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    with open(TEST_S1_PATH, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
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

    total_s1 = sum(len(d) for d in s1_by_country.values())
    print(f"  Ingested {total_s1:,} Source 1 entities across {len(s1_by_country)} countries:")
    for ctry, d in sorted(s1_by_country.items()):
        print(f"    - {ctry:10s}: {len(d):,} entities ({len(d)/total_s1*100:.1f}%)")

    # 3. Stream Inference by Country
    print("\n[3/4] Executing Country-Partitioned Inference & Output Generation...")
    os.makedirs(os.path.dirname(args.matching_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.candidate_out), exist_ok=True)

    with open(args.matching_out, "w", encoding="utf-8", newline="") as f_match, \
         open(args.candidate_out, "w", encoding="utf-8", newline="") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in ["France", "US", "India"]:
            if country in s1_by_country:
                process_country_partition(
                    country=country,
                    s1_records=s1_by_country[country],
                    model=model,
                    tau=tau,
                    extractor=extractor,
                    out_matching_file=f_match,
                    out_candidate_file=f_cand,
                    batch_size=args.batch_size
                )

    print(f"\nOutputs generated:")
    print(f"  - Matching TSV  : {args.matching_out}")
    print(f"  - Candidate TSV : {args.candidate_out}")

    # 4. Official Validator Execution
    if args.validate and os.path.isfile(os.path.join(BASE_DIR, "utils", "validate_submission.py")):
        print("\n[4/4] Running Official validate_submission.py...")
        import subprocess
        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "utils", "validate_submission.py"),
            "--matching", args.matching_out,
            "--candidate", args.candidate_out,
            "--test-dir", os.path.join(BASE_DIR, "dataset", "test")
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print("VALIDATION SUCCESS: Submission files are 100% compliant!")
        else:
            print(f"VALIDATION WARNING: Validator returned exit code {res.returncode}")

    print(f"\nTotal Pipeline Execution Time: {time.time() - t_master:.2f}s ({(time.time() - t_master)/60:.2f} min)")


if __name__ == "__main__":
    main()
