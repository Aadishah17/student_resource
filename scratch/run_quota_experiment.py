"""
ML Challenge 2026: Business Entity Resolution
Validation-Only Experiment: Shared Quota (Config A) vs Balanced Quota (Config B)

This experiment evaluates the candidate generation quota configuration on the
held-out validation cohort (N=2,000 S1 entities from dataset/processed/val_data.npz).

Configurations Evaluated:
A) Current Shared Quota:
   MAX_S1_CANDS = 150 (S2 streamed first, saturating candidate pool before S3)
B) Balanced Quota:
   MAX_S2_CANDS = 75, MAX_S3_CANDS = 75 (total = 150)
Additional frontier points:
   70/80, 80/70, 60/90, 90/60, 100/100, Unbounded

Metrics Measured:
1. Candidate recall against ground-truth links (overall, S2, S3)
2. Macro F0.5 at tau = 0.72 using champion XGBoost model
3. Pairwise precision and pairwise recall
4. Singleton accuracy
5. Average and P95 candidates/S1
6. Candidate S2/S3 percentage
7. Number of true links recovered that were previously blocked out
8. Runtime and peak RAM impact

Strict Safety:
- VALIDATION ONLY: reads from dataset/train/, does NOT touch output/
- Does NOT alter active production inference run or checkpoints
- Does NOT alter models or thresholds
"""

import os
import sys
import time
import json
import csv
import pickle
import gc
import psutil
from collections import defaultdict, Counter, OrderedDict
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
import xgboost as xgb

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
from inference_v2 import extract_features_v2_multiprocess

# Constants matching production Configuration H
MAX_S1_ADDR_TOKEN_FREQ = 5
MAX_S1_NGRAM_FREQ = 2
MAX_TARGET_ADDR_TOKEN_FREQ = DEFAULT_MAX_ADDR_TOKEN_FREQ  # 80
MAX_TARGET_NGRAM_FREQ = 50
TAU = 0.72


def get_mem_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def compute_macro_f05(
    s1_ids: List[str],
    s1_preds: Dict[str, Set[str]],
    val_gt: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Computes exact Macro F0.5, pairwise precision, pairwise recall, and singleton accuracy."""
    s1_scores = []
    correct_sing = 0
    total_sing = 0
    tp_pairs = 0
    fp_pairs = 0
    fn_pairs = 0

    for s in s1_ids:
        preds = s1_preds.get(s, set())
        truth = val_gt[s]

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
        "macro_f05": macro_f05,
        "precision": prec,
        "recall": rec,
        "singleton_accuracy": sing_acc,
        "tp": tp_pairs,
        "fp": fp_pairs,
        "fn": fn_pairs
    }


def main():
    start_time = time.time()
    print("=" * 80)
    print("VALIDATION EXPERIMENT: CANDIDATE GENERATION QUOTA ABLATION")
    print("Evaluating Shared Quota (Config A) vs Balanced Quota (Config B)")
    print("=" * 80)

    # 1. Load Validation S1 Cohort and Ground Truth
    val_npz_path = os.path.join(BASE_DIR, "dataset", "processed", "val_data.npz")
    print(f"\n[1/5] Loading held-out validation cohort from {val_npz_path}...")
    val_npz = np.load(val_npz_path, allow_pickle=True)
    val_s1_ids = sorted(list(set(val_npz["s1_ids"])))
    val_s1_set = set(val_s1_ids)
    print(f"  Loaded {len(val_s1_ids):,} unique validation S1 IDs.")

    # Load Ground Truth
    gt_path = os.path.join(BASE_DIR, "dataset", "train", "train_ground_truth.tsv")
    val_gt: Dict[str, Set[str]] = {s: set() for s in val_s1_ids}
    with open(gt_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in val_gt:
                matches = set(x.strip() for x in row[1].split(",") if x.strip())
                val_gt[row[0]] = matches

    total_gt_links = sum(len(m) for m in val_gt.values())
    gt_s2_links = sum(1 for m in val_gt.values() for x in m if x.startswith("S2-"))
    gt_s3_links = sum(1 for m in val_gt.values() for x in m if x.startswith("S3-"))
    gt_singletons = sum(1 for m in val_gt.values() if len(m) == 0)

    print(f"  Ground Truth Links: {total_gt_links:,}")
    print(f"    - S2 Links: {gt_s2_links:,} ({gt_s2_links/total_gt_links*100:.2f}%)")
    print(f"    - S3 Links: {gt_s3_links:,} ({gt_s3_links/total_gt_links*100:.2f}%)")
    print(f"    - Singletons: {gt_singletons:,} ({gt_singletons/len(val_s1_ids)*100:.2f}%)")

    # Load Source 1 records
    s1_path = os.path.join(BASE_DIR, "dataset", "train", "train_source1.tsv")
    val_s1_records: Dict[str, Dict[str, Any]] = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            if eid in val_s1_set:
                norm = normalize_name(name)
                val_s1_records[eid] = {
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
                if len(val_s1_records) == len(val_s1_set):
                    break

    countries = sorted(list(set(r["country"] for r in val_s1_records.values())))
    print(f"  Validation Countries: {countries}")
    for c in countries:
        cnt = sum(1 for r in val_s1_records.values() if r["country"] == c)
        print(f"    - {c}: {cnt:,} S1 entities")

    # 2. Build Inverted Indexes Partitioned by Country
    print("\n[2/5] Building inverted indexes for validation S1 entities...")
    country_indexes: Dict[str, Dict[str, Dict[Any, List[str]]]] = {}
    country_key_freq: Dict[str, Counter] = {c: Counter() for c in countries}

    for ctry in countries:
        ctry_s1 = {eid: r for eid, r in val_s1_records.items() if r["country"] == ctry}
        s1_addr_counts = Counter()
        s1_ngram_counts = Counter()

        for raw in ctry_s1.values():
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
        for eid, raw in ctry_s1.items():
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

            if norm_addr:
                rare_addr_toks = [t for t in norm_addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and s1_addr_counts[t] <= MAX_S1_ADDR_TOKEN_FREQ]
                for t in rare_addr_toks[:3]:
                    c_idx["addr_token"][t].append(eid)

            eff_nosuff = trans_nosuff if (has_ind and trans_nosuff) else nosuff
            comp_eff = compact_string(eff_nosuff) if eff_nosuff else ""
            if len(comp_eff) >= 5:
                rare_grams = [comp_eff[j:j+5] for j in range(len(comp_eff) - 4) if s1_ngram_counts[comp_eff[j:j+5]] <= MAX_S1_NGRAM_FREQ]
                for g in rare_grams:
                    c_idx["char_ngram"][g].append(eid)
            elif len(comp_eff) == 4 and s1_ngram_counts[comp_eff] <= MAX_S1_NGRAM_FREQ:
                c_idx["char_ngram"][comp_eff].append(eid)

        country_indexes[ctry] = c_idx
        print(f"  Index built for {ctry}: RAM {get_mem_mb():.2f} MB")

    # 3. Stream Target Files (S2 then S3) or Load from Cache
    cache_path = os.path.join(BASE_DIR, "scratch", "val_streaming_cache.pkl")
    if os.path.exists(cache_path):
        print(f"\n[3/5] Loading cached streaming candidates from {cache_path}...")
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
            s2_candidates = cache_data["s2_candidates"]
            s3_candidates = cache_data["s3_candidates"]
            raw_targets = cache_data["raw_targets"]
            unbounded_hits = cache_data["unbounded_hits"]
            t_stream_total = cache_data.get("t_stream_total", 560.2)
        print(f"  [CACHE HIT] Loaded {len(s2_candidates):,} S1 candidate profiles, {len(raw_targets):,} unique targets. RAM: {get_mem_mb():.2f} MB")
    else:
        print("\n[3/5] Streaming target databases (train_source2.tsv, train_source3.tsv)...")
        s2_candidates: Dict[str, OrderedDict[str, Set[str]]] = {eid: OrderedDict() for eid in val_s1_ids}
        s3_candidates: Dict[str, OrderedDict[str, Set[str]]] = {eid: OrderedDict() for eid in val_s1_ids}
        raw_targets: Dict[str, Tuple[str, str, str]] = {}
        unbounded_hits: Dict[str, Set[str]] = {eid: set() for eid in val_s1_ids}

        target_files = [
            ("Source 2", os.path.join(BASE_DIR, "dataset", "train", "train_source2.tsv")),
            ("Source 3", os.path.join(BASE_DIR, "dataset", "train", "train_source3.tsv"))
        ]

        t_stream_start = time.time()
        MAX_RECORD_CANDS = 150  # Captures up to 150 for S2 and 150 for S3 to allow evaluating all quotas

        for source_name, fpath in target_files:
            is_s2 = "source2" in fpath.lower()
            t0 = time.time()
            row_count = 0
            hit_count = 0
            print(f"  Streaming {source_name} from {fpath}...")

            with open(fpath, "r", encoding="utf-8") as f:
                r = csv.reader(f, delimiter="\t")
                next(r)
                for row in r:
                    row_count += 1
                    if row_count % 1000000 == 0:
                        print(f"    ... {row_count:,} rows streamed ({time.time()-t0:.1f}s, hits so far: {hit_count:,})", flush=True)

                    ccountry = row[3]
                    if ccountry not in country_indexes:
                        continue

                    c_idx = country_indexes[ccountry]
                    key_freq = country_key_freq[ccountry]
                    cid, cname, caddr = row[0], row[1], row[2]

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
                        k = (ccountry, "skel", cskel[:5])
                        key_freq[k] += 1
                        if key_freq[k] <= 200:
                            for s1 in c_idx["consonant_skel"][cskel[:5]]: hits.append((s1, "blocked_consonant_skeleton"))
                    if cnorm_addr and len(cnorm) >= 2:
                        for t in [x for x in cnorm_addr.split() if len(x) >= 4 and not x.isdigit()][:3]:
                            if (t, cnorm[:2]) in c_idx["addr_name"]:
                                k = (ccountry, "addr_name", (t, cnorm[:2]))
                                key_freq[k] += 1
                                if key_freq[k] <= 200:
                                    for s1 in c_idx["addr_name"][(t, cnorm[:2])]: hits.append((s1, "blocked_address_name_combo"))

                    if cnorm_addr:
                        for t in [x for x in cnorm_addr.split() if len(x) >= 4 and not x.isdigit() and x not in ADDR_STOPWORDS][:3]:
                            if t in c_idx["addr_token"]:
                                k = (ccountry, "addr_token", t)
                                key_freq[k] += 1
                                if key_freq[k] <= MAX_TARGET_ADDR_TOKEN_FREQ:
                                    for s1 in c_idx["addr_token"][t]: hits.append((s1, "blocked_address_token"))

                    c_eff_nosuff = ctrans_nosuff if (chas_ind and ctrans_nosuff) else cnosuff
                    c_comp_eff = compact_string(c_eff_nosuff) if c_eff_nosuff else ""
                    if len(c_comp_eff) >= 5:
                        c_grams = set(c_comp_eff[j:j+5] for j in range(len(c_comp_eff) - 4))
                        for g in c_grams:
                            if g in c_idx["char_ngram"]:
                                k = (ccountry, "char_ngram", g)
                                key_freq[k] += 1
                                if key_freq[k] <= MAX_TARGET_NGRAM_FREQ:
                                    for s1 in c_idx["char_ngram"][g]: hits.append((s1, "blocked_char_ngram"))
                    elif len(c_comp_eff) == 4 and c_comp_eff in c_idx["char_ngram"]:
                        k = (ccountry, "char_ngram", c_comp_eff)
                        key_freq[k] += 1
                        if key_freq[k] <= MAX_TARGET_NGRAM_FREQ:
                            for s1 in c_idx["char_ngram"][c_comp_eff]: hits.append((s1, "blocked_char_ngram"))

                    if hits:
                        for s1, rule_name in hits:
                            hit_count += 1
                            unbounded_hits[s1].add(cid)
                            target_dict = s2_candidates[s1] if is_s2 else s3_candidates[s1]
                            if cid in target_dict:
                                target_dict[cid].add(rule_name)
                            elif len(target_dict) < MAX_RECORD_CANDS:
                                target_dict[cid] = {rule_name}
                                if cid not in raw_targets:
                                    raw_targets[cid] = (cname, caddr, ccountry)

            print(f"  Finished {source_name}: {row_count:,} rows in {time.time()-t0:.1f}s, hits: {hit_count:,}, unique targets stored: {len(raw_targets):,}")

        t_stream_total = time.time() - t_stream_start
        print(f"  Streaming phase complete in {t_stream_total:.1f}s. RAM: {get_mem_mb():.2f} MB")

        # Save to cache file
        print(f"  [CACHE] Saving streaming results to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump({
                "s2_candidates": s2_candidates,
                "s3_candidates": s3_candidates,
                "raw_targets": raw_targets,
                "unbounded_hits": unbounded_hits,
                "t_stream_total": t_stream_total
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [CACHE] Saved {os.path.getsize(cache_path)/(1024*1024):.2f} MB cache.")

    # 4. Form Candidate Sets for Config A, Config B, and Quota Frontier
    # Config A: Shared 150 (all S2 up to 150, then S3 fills up to 150)
    # Config B: Balanced 75/75 (first 75 S2, first 75 S3)
    def build_candidates_for_quota(q_s2: int, q_s3: int, shared_limit: Optional[int] = None):
        cands = defaultdict(dict)
        for s1 in val_s1_ids:
            s2_items = list(s2_candidates[s1].items())
            s3_items = list(s3_candidates[s1].items())

            if shared_limit is not None:
                # S2 first streaming up to shared_limit
                s2_take = s2_items[:shared_limit]
                rem = shared_limit - len(s2_take)
                s3_take = s3_items[:rem] if rem > 0 else []
            else:
                s2_take = s2_items[:q_s2]
                s3_take = s3_items[:q_s3]

            for cid, rules in s2_take:
                cands[s1][cid] = rules
            for cid, rules in s3_take:
                cands[s1][cid] = rules
        return cands

    configs = {
        "Config A (Shared 150, Current S2-First)": build_candidates_for_quota(150, 150, shared_limit=150),
        "Config B (Balanced 75/75)": build_candidates_for_quota(75, 75),
        "Quota 70/80 (S2=70, S3=80)": build_candidates_for_quota(70, 80),
        "Quota 80/70 (S2=80, S3=70)": build_candidates_for_quota(80, 70),
        "Quota 60/90 (S2=60, S3=90)": build_candidates_for_quota(60, 90),
        "Quota 90/60 (S2=90, S3=60)": build_candidates_for_quota(90, 60),
        "Quota 100/100 (Total 200)": build_candidates_for_quota(100, 100),
    }

    # Measure Candidate Statistics and Recalls
    cand_metrics = {}
    for cfg_name, cands_dict in configs.items():
        total_cands = sum(len(c) for c in cands_dict.values())
        cands_per_s1 = [len(cands_dict[s1]) for s1 in val_s1_ids]
        mean_cands = float(np.mean(cands_per_s1))
        p95_cands = float(np.percentile(cands_per_s1, 95))
        max_cands = int(np.max(cands_per_s1))

        # S2 vs S3 candidate counts
        n_s2_cands = sum(1 for c in cands_dict.values() for cid in c if cid.startswith("S2-"))
        n_s3_cands = sum(1 for c in cands_dict.values() for cid in c if cid.startswith("S3-"))
        pct_s2 = (n_s2_cands / total_cands * 100) if total_cands > 0 else 0.0
        pct_s3 = (n_s3_cands / total_cands * 100) if total_cands > 0 else 0.0

        # Candidate recall against ground truth
        matched_gt_total = 0
        matched_gt_s2 = 0
        matched_gt_s3 = 0

        for s1 in val_s1_ids:
            gt_set = val_gt[s1]
            cand_set = set(cands_dict[s1].keys())
            matched = gt_set & cand_set
            matched_gt_total += len(matched)
            matched_gt_s2 += sum(1 for x in matched if x.startswith("S2-"))
            matched_gt_s3 += sum(1 for x in matched if x.startswith("S3-"))

        rec_total = matched_gt_total / total_gt_links if total_gt_links > 0 else 0.0
        rec_s2 = matched_gt_s2 / gt_s2_links if gt_s2_links > 0 else 0.0
        rec_s3 = matched_gt_s3 / gt_s3_links if gt_s3_links > 0 else 0.0

        cand_metrics[cfg_name] = {
            "total_cands": total_cands,
            "mean_cands": mean_cands,
            "p95_cands": p95_cands,
            "max_cands": max_cands,
            "n_s2": n_s2_cands,
            "n_s3": n_s3_cands,
            "pct_s2": pct_s2,
            "pct_s3": pct_s3,
            "matched_gt_total": matched_gt_total,
            "matched_gt_s2": matched_gt_s2,
            "matched_gt_s3": matched_gt_s3,
            "recall_total": rec_total,
            "recall_s2": rec_s2,
            "recall_s3": rec_s3
        }

    # Also compute unbounded candidate recall (theoretical maximum of Configuration H)
    unbounded_matched_total = sum(len(val_gt[s1] & unbounded_hits[s1]) for s1 in val_s1_ids)
    unbounded_matched_s2 = sum(1 for s1 in val_s1_ids for x in (val_gt[s1] & unbounded_hits[s1]) if x.startswith("S2-"))
    unbounded_matched_s3 = sum(1 for s1 in val_s1_ids for x in (val_gt[s1] & unbounded_hits[s1]) if x.startswith("S3-"))
    cand_metrics["Unbounded (No Cap)"] = {
        "total_cands": sum(len(c) for c in unbounded_hits.values()),
        "mean_cands": float(np.mean([len(unbounded_hits[s]) for s in val_s1_ids])),
        "p95_cands": float(np.percentile([len(unbounded_hits[s]) for s in val_s1_ids], 95)),
        "max_cands": int(np.max([len(unbounded_hits[s]) for s in val_s1_ids])),
        "n_s2": sum(1 for c in unbounded_hits.values() for cid in c if cid.startswith("S2-")),
        "n_s3": sum(1 for c in unbounded_hits.values() for cid in c if cid.startswith("S3-")),
        "pct_s2": sum(1 for c in unbounded_hits.values() for cid in c if cid.startswith("S2-")) / sum(len(c) for c in unbounded_hits.values()) * 100,
        "pct_s3": sum(1 for c in unbounded_hits.values() for cid in c if cid.startswith("S3-")) / sum(len(c) for c in unbounded_hits.values()) * 100,
        "matched_gt_total": unbounded_matched_total,
        "matched_gt_s2": unbounded_matched_s2,
        "matched_gt_s3": unbounded_matched_s3,
        "recall_total": unbounded_matched_total / total_gt_links,
        "recall_s2": unbounded_matched_s2 / gt_s2_links,
        "recall_s3": unbounded_matched_s3 / gt_s3_links
    }

    print("\n--- CANDIDATE GENERATION RECALL & DISTRIBUTION COMPARISON ---")
    print(f"{'Configuration':<38} | {'Recall':<7} | {'S2 Rec':<7} | {'S3 Rec':<7} | {'Mean Cands':<10} | {'P95 Cands':<9} | {'% S2':<6} | {'% S3':<6}")
    print("-" * 105)
    for name, m in cand_metrics.items():
        print(f"{name:<38} | {m['recall_total']*100:6.2f}% | {m['recall_s2']*100:6.2f}% | {m['recall_s3']*100:6.2f}% | {m['mean_cands']:10.1f} | {m['p95_cands']:9.1f} | {m['pct_s2']:5.1f}% | {m['pct_s3']:5.1f}%")

    # Links recovered in Config B that were blocked out in Config A
    cands_A = configs["Config A (Shared 150, Current S2-First)"]
    cands_B = configs["Config B (Balanced 75/75)"]
    recovered_gt_links = []
    blocked_in_B_links = []
    for s1 in val_s1_ids:
        gt_set = val_gt[s1]
        set_A = set(cands_A[s1].keys())
        set_B = set(cands_B[s1].keys())
        for g in gt_set:
            if g in set_B and g not in set_A:
                recovered_gt_links.append((s1, g))
            elif g in set_A and g not in set_B:
                blocked_in_B_links.append((s1, g))

    print(f"\nLink Recovery Analysis (Config A vs Config B):")
    print(f"  - True links recovered by Config B that were blocked in Config A: {len(recovered_gt_links):,}")
    rec_s2 = sum(1 for _, g in recovered_gt_links if g.startswith("S2-"))
    rec_s3 = sum(1 for _, g in recovered_gt_links if g.startswith("S3-"))
    print(f"    * Recovered S2 links: {rec_s2:,}")
    print(f"    * Recovered S3 links: {rec_s3:,}")
    print(f"  - True links in Config A dropped in Config B: {len(blocked_in_B_links):,}")
    drop_s2 = sum(1 for _, g in blocked_in_B_links if g.startswith("S2-"))
    drop_s3 = sum(1 for _, g in blocked_in_B_links if g.startswith("S3-"))
    print(f"    * Dropped S2 links: {drop_s2:,}")
    print(f"    * Dropped S3 links: {drop_s3:,}")
    print(f"  - Net True Links Gained: {len(recovered_gt_links) - len(blocked_in_B_links):+,} ({((len(recovered_gt_links) - len(blocked_in_B_links))/total_gt_links)*100:+.2f}% absolute recall gain)")

    # 5. Feature Extraction and Champion Model Scoring
    print("\n[4/5] Extracting 96 features and scoring with champion XGBoost model...")
    # Gather union of pairs across all configurations to compute features once
    union_pairs_dict: Dict[Tuple[str, str], Set[str]] = {}
    for cfg_dict in configs.values():
        for s1, c_map in cfg_dict.items():
            for cid, rules in c_map.items():
                k = (s1, cid)
                if k not in union_pairs_dict:
                    union_pairs_dict[k] = set(rules)
                else:
                    union_pairs_dict[k].update(rules)

    flat_pairs = sorted([(s1, cid, rules) for (s1, cid), rules in union_pairs_dict.items()])
    print(f"  Total unique candidate pairs to score across all configurations: {len(flat_pairs):,}")

    t_feat_start = time.time()
    s1_enriched = {eid: enrich_record_for_features(val_s1_records[eid]) for eid in val_s1_ids}
    cand_ids_needed = set(cid for _, cid, _ in flat_pairs)
    raw_targets_map = {cid: raw_targets[cid] for cid in cand_ids_needed if cid in raw_targets}

    # Multiprocess feature extraction using 8 workers
    feat_mat = extract_features_v2_multiprocess(
        flat_pairs, s1_enriched, raw_targets_map, num_workers=8
    )
    t_feat_total = time.time() - t_feat_start
    print(f"  Feature extraction complete in {t_feat_total:.2f}s ({len(flat_pairs)/t_feat_total:.0f} pairs/s). RAM: {get_mem_mb():.2f} MB")

    # Load Model and Score
    model_path = os.path.join(SRC_DIR, "..", "models", "best_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    t_score_start = time.time()
    probs = model.predict_proba(feat_mat)[:, 1]
    t_score_total = time.time() - t_score_start
    print(f"  Model scoring complete in {t_score_total:.2f}s ({len(flat_pairs)/t_score_total:.0f} pairs/s). RAM: {get_mem_mb():.2f} MB")

    pair_probs: Dict[Tuple[str, str], float] = {
        (p[0], p[1]): float(prob) for p, prob in zip(flat_pairs, probs)
    }

    # Free memory
    del feat_mat, probs
    gc.collect()

    # 6. Evaluate Macro F0.5, Precision, Recall, Singleton Accuracy
    print("\n[5/5] Evaluating end-to-end resolution performance at tau = 0.72...")
    model_eval_results = {}
    recovered_predicted_count = 0
    for s1, g in recovered_gt_links:
        if pair_probs.get((s1, g), 0.0) >= TAU:
            recovered_predicted_count += 1

    for cfg_name, cands_dict in configs.items():
        s1_preds: Dict[str, Set[str]] = defaultdict(set)
        for s1, c_map in cands_dict.items():
            for cid in c_map.keys():
                p = pair_probs.get((s1, cid), 0.0)
                if p >= TAU:
                    s1_preds[s1].add(cid)

        metrics = compute_macro_f05(val_s1_ids, s1_preds, val_gt)
        # Add candidate recall info
        metrics["cand_recall"] = cand_metrics[cfg_name]["recall_total"]
        metrics["cand_s2_rec"] = cand_metrics[cfg_name]["recall_s2"]
        metrics["cand_s3_rec"] = cand_metrics[cfg_name]["recall_s3"]
        metrics["mean_cands"] = cand_metrics[cfg_name]["mean_cands"]
        metrics["p95_cands"] = cand_metrics[cfg_name]["p95_cands"]
        metrics["pct_s2"] = cand_metrics[cfg_name]["pct_s2"]
        metrics["pct_s3"] = cand_metrics[cfg_name]["pct_s3"]
        model_eval_results[cfg_name] = metrics

    print("\n" + "=" * 115)
    print(f"{'Configuration':<38} | {'Macro F0.5':<10} | {'Pair Prec':<9} | {'Pair Rec':<9} | {'Sing Acc':<9} | {'Cand Rec':<9} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
    print("=" * 115)
    for name, m in model_eval_results.items():
        print(f"{name:<38} | {m['macro_f05']*100:9.2f}% | {m['precision']*100:8.2f}% | {m['recall']*100:8.2f}% | {m['singleton_accuracy']*100:8.2f}% | {m['cand_recall']*100:8.2f}% | {m['tp']:5d} | {m['fp']:5d} | {m['fn']:5d}")
    print("=" * 115)

    print(f"\nRecovered Links Model Validation:")
    print(f"  - Total true links recovered in candidate pool: {len(recovered_gt_links):,}")
    print(f"  - True links recovered AND predicted positive by model (p >= {TAU}): {recovered_predicted_count:,} ({recovered_predicted_count/len(recovered_gt_links)*100:.1f}%)")

    # Detailed Comparison between Config A and Config B
    mA = model_eval_results["Config A (Shared 150, Current S2-First)"]
    mB = model_eval_results["Config B (Balanced 75/75)"]

    print("\n" + "=" * 80)
    print("EXECUTIVE SUMMARY: CONFIG A vs CONFIG B")
    print("=" * 80)
    print(f"Metric                               | Config A (Current) | Config B (Balanced 75/75) | Absolute Delta")
    print("-" * 80)
    print(f"Candidate Recall (Overall)           | {mA['cand_recall']*100:17.2f}% | {mB['cand_recall']*100:23.2f}% | {mB['cand_recall']*100 - mA['cand_recall']*100:+13.2f}%")
    print(f"Candidate Recall (Source 2)          | {mA['cand_s2_rec']*100:17.2f}% | {mB['cand_s2_rec']*100:23.2f}% | {mB['cand_s2_rec']*100 - mA['cand_s2_rec']*100:+13.2f}%")
    print(f"Candidate Recall (Source 3)          | {mA['cand_s3_rec']*100:17.2f}% | {mB['cand_s3_rec']*100:23.2f}% | {mB['cand_s3_rec']*100 - mA['cand_s3_rec']*100:+13.2f}%")
    print(f"Macro F0.5 (Threshold 0.72)          | {mA['macro_f05']*100:17.2f}% | {mB['macro_f05']*100:23.2f}% | {mB['macro_f05']*100 - mA['macro_f05']*100:+13.2f}%")
    print(f"Pairwise Precision                   | {mA['precision']*100:17.2f}% | {mB['precision']*100:23.2f}% | {mB['precision']*100 - mA['precision']*100:+13.2f}%")
    print(f"Pairwise Recall                      | {mA['recall']*100:17.2f}% | {mB['recall']*100:23.2f}% | {mB['recall']*100 - mA['recall']*100:+13.2f}%")
    print(f"Singleton Accuracy                   | {mA['singleton_accuracy']*100:17.2f}% | {mB['singleton_accuracy']*100:23.2f}% | {mB['singleton_accuracy']*100 - mA['singleton_accuracy']*100:+13.2f}%")
    print(f"True Positive Pairs (TP)             | {mA['tp']:18d} | {mB['tp']:25d} | {mB['tp'] - mA['tp']:+15d}")
    print(f"False Positive Pairs (FP)            | {mA['fp']:18d} | {mB['fp']:25d} | {mB['fp'] - mA['fp']:+15d}")
    print(f"False Negative Pairs (FN)            | {mA['fn']:18d} | {mB['fn']:25d} | {mB['fn'] - mA['fn']:+15d}")
    print(f"Average Candidates per S1            | {mA['mean_cands']:18.1f} | {mB['mean_cands']:25.1f} | {mB['mean_cands'] - mA['mean_cands']:+15.1f}")
    print(f"P95 Candidates per S1                | {mA['p95_cands']:18.1f} | {mB['p95_cands']:25.1f} | {mB['p95_cands'] - mA['p95_cands']:+15.1f}")
    print(f"Candidate S2 Percentage              | {mA['pct_s2']:17.1f}% | {mB['pct_s2']:23.1f}% | {mB['pct_s2'] - mA['pct_s2']:+13.1f}%")
    print(f"Candidate S3 Percentage              | {mA['pct_s3']:17.1f}% | {mB['pct_s3']:23.1f}% | {mB['pct_s3'] - mA['pct_s3']:+13.1f}%")
    print("-" * 80)
    print(f"Peak RAM: {get_mem_mb():.2f} MB")
    print(f"Total Experiment Time: {time.time()-start_time:.1f}s")
    print("=" * 80)

    # Save validation results to JSON for artifact generation
    results_json_path = os.path.join(BASE_DIR, "analysis", "validation_quota_experiment_results.json")
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "cand_metrics": cand_metrics,
            "model_eval_results": model_eval_results,
            "link_recovery": {
                "total_recovered_in_candidates": len(recovered_gt_links),
                "recovered_s2": rec_s2,
                "recovered_s3": rec_s3,
                "dropped_in_candidates": len(blocked_in_B_links),
                "dropped_s2": drop_s2,
                "dropped_s3": drop_s3,
                "recovered_predicted_positive": recovered_predicted_count
            },
            "peak_ram_mb": get_mem_mb(),
            "total_time_seconds": time.time() - start_time
        }, f, indent=2)
    print(f"Saved results to {results_json_path}")


if __name__ == "__main__":
    main()
