"""
ML Challenge 2026: Business Entity Resolution
Module: inference_v2.py

High-performance V2 inference engine with multi-process feature extraction.
Optimized for multi-core systems (24 logical cores) with bounded RAM footprint:
- Preserves exact 96-feature schema and numerical definitions from PairwiseFeatureExtractor.
- Partitions candidate-pair feature extraction across N worker processes.
- Eliminates memory bloat: avoids eager 15 GB pre-enrichment of 500k+ targets.
- Passes only compact raw target tuples (name, address, country) per worker slice.
- Uses shared memory (multiprocessing.RawArray) for zero-IPC feature matrix assembly.
- Workers enrich targets on-demand with bounded local caching (<400 MB RAM per worker).
- Concatenates feature chunks in exact, deterministic candidate order.
- Runs champion 50k XGBoost model unchanged.
"""

import os
import sys
import time
import math
import psutil
import multiprocessing as mp
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import xgboost as xgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
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

MAX_LOCAL_TARGET_CACHE = 40000


def _worker_feature_extraction(
    worker_id: int,
    start_idx: int,
    end_idx: int,
    worker_pairs: List[Tuple[str, str, Optional[Set[str]]]],
    worker_s1: Dict[str, Dict[str, Any]],
    worker_raw_targets: Dict[str, Tuple[str, str, str]],
    raw_arr: Any,
    total_pairs: int,
    num_features: int
):
    """
    Worker process task:
    Extracts 96-feature vectors for worker_pairs and writes directly into
    the shared float32 memory block raw_arr at indices [start_idx : end_idx].
    """
    extractor = PairwiseFeatureExtractor()
    # Map shared RawArray to numpy 2D array view
    shared_mat = np.frombuffer(raw_arr, dtype=np.float32).reshape((total_pairs, num_features))

    target_cache: Dict[str, Dict[str, Any]] = {}

    for i, (s1_id, cand_id, ev_rules) in enumerate(worker_pairs):
        s1_rec = worker_s1[s1_id]

        cand_rec = target_cache.get(cand_id)
        if cand_rec is None:
            name, addr, ctry = worker_raw_targets[cand_id]
            cand_rec = enrich_record_for_features({
                "entity_id": cand_id,
                "name": name,
                "address": addr,
                "country": ctry
            })
            if len(target_cache) >= MAX_LOCAL_TARGET_CACHE:
                target_cache.clear()
            target_cache[cand_id] = cand_rec

        row_idx = start_idx + i
        shared_mat[row_idx, :] = extractor.extract_features_vector(s1_rec, cand_rec, ev_rules)


def extract_features_v2_multiprocess(
    flat_pairs: List[Tuple[str, str, Optional[Set[str]]]],
    s1_enriched: Dict[str, Dict[str, Any]],
    needed_raw_targets: Dict[str, Tuple[str, str, str]],
    num_workers: int = 4
) -> np.ndarray:
    """
    Extracts the dense (N, 96) float32 feature matrix across num_workers processes.
    Writes directly into a shared RawArray to avoid IPC serialization overhead.
    """
    n_pairs = len(flat_pairs)
    if n_pairs == 0:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)

    # 1. Allocate shared float32 buffer
    raw_arr = mp.RawArray('f', n_pairs * NUM_FEATURES)

    if num_workers <= 1:
        # Run in-process with the same zero-IPC RawArray
        _worker_feature_extraction(
            worker_id=0,
            start_idx=0,
            end_idx=n_pairs,
            worker_pairs=flat_pairs,
            worker_s1=s1_enriched,
            worker_raw_targets=needed_raw_targets,
            raw_arr=raw_arr,
            total_pairs=n_pairs,
            num_features=NUM_FEATURES
        )
        return np.frombuffer(raw_arr, dtype=np.float32).reshape((n_pairs, NUM_FEATURES)).copy()

    # 2. Partition flat_pairs into contiguous slices
    chunk_size = (n_pairs + num_workers - 1) // num_workers
    procs: List[mp.Process] = []

    for w in range(num_workers):
        start_idx = w * chunk_size
        end_idx = min(start_idx + chunk_size, n_pairs)
        if start_idx >= end_idx:
            continue

        worker_pairs = flat_pairs[start_idx:end_idx]

        # Extract only the S1 and target IDs needed by this worker
        needed_s1_ids = {p[0] for p in worker_pairs}
        needed_cand_ids = {p[1] for p in worker_pairs}

        worker_s1 = {eid: s1_enriched[eid] for eid in needed_s1_ids if eid in s1_enriched}
        worker_raw_targets = {cid: needed_raw_targets[cid] for cid in needed_cand_ids if cid in needed_raw_targets}

        p = mp.Process(
            target=_worker_feature_extraction,
            args=(
                w,
                start_idx,
                end_idx,
                worker_pairs,
                worker_s1,
                worker_raw_targets,
                raw_arr,
                n_pairs,
                NUM_FEATURES
            )
        )
        p.start()
        procs.append(p)

    # 3. Wait for all workers to complete
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    # 4. Return as numpy array
    return np.frombuffer(raw_arr, dtype=np.float32).reshape((n_pairs, NUM_FEATURES)).copy()


def score_pairs_v2(
    model: xgb.XGBClassifier,
    feat_mat: np.ndarray,
    flat_pairs: List[Tuple[str, str, Optional[Set[str]]]],
    batch_s1_ids: List[str],
    tau: float = 0.72,
    chunk_predict_size: int = 50000
) -> Dict[str, List[str]]:
    """
    Runs XGBoost predict_proba on the feature matrix and groups matches by s1_id.
    """
    n_pairs = len(flat_pairs)
    matches_by_s1: Dict[str, List[str]] = {eid: [] for eid in batch_s1_ids}
    if n_pairs == 0:
        return matches_by_s1

    for c_start in range(0, n_pairs, chunk_predict_size):
        c_end = min(c_start + chunk_predict_size, n_pairs)
        chunk_mat = feat_mat[c_start:c_end]
        probs = model.predict_proba(chunk_mat)[:, 1]

        for i in range(c_end - c_start):
            if probs[i] >= tau:
                s1_id, cand_id, _ = flat_pairs[c_start + i]
                matches_by_s1[s1_id].append(cand_id)

    return matches_by_s1
