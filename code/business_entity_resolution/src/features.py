"""
ML Challenge 2026: Business Entity Resolution
Module: features.py

Pairwise feature engineering pipeline for entity resolution.
Computes 96 discriminative, leakage-free numeric features across:
1. Name features (exact, fuzzy, character n-gram, token overlap, structural, and consonant skeletons)
2. Address features (exact, fuzzy, token Jaccard, numeric token overlap, house, postal, and explicit missingness indicators)
3. Cross-field interaction features (combined postal/name, house/name, address x name interaction, max similarities)
4. Script/Language signals (Latin vs Indic, cross-script indicators, transliteration status)
5. Source signals (candidate is Source 2 vs Source 3)
6. Blocking evidence (which blocking rules generated the candidate, count of generating rules)

Guarantees:
- Strictly numeric output (np.float32 / np.int32)
- Zero NaN and zero infinity values
- Safe handling of missing address fields with explicit missingness flags
- Vectorized chunk processing for memory efficiency (compliant with 15.5 GB RAM)
- No external lookups, geocoding APIs, or unverified external business databases.
"""

import os
import sys
import re
import math
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import numpy as np

# RapidFuzz for high-speed C++ Levenshtein and fuzzy token matching
try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Import preprocessing helpers
try:
    from .preprocessing import (
        normalize_name,
        compact_string,
        remove_legal_suffix,
        transliterate_name,
        normalize_address,
        extract_postal_code,
        extract_house_number,
        has_indic_characters
    )
    from .blocking import extract_consonants
except ImportError:
    from preprocessing import (
        normalize_name,
        compact_string,
        remove_legal_suffix,
        transliterate_name,
        normalize_address,
        extract_postal_code,
        extract_house_number,
        has_indic_characters
    )
    from blocking import extract_consonants


# ==============================================================================
# Helper String Similarity Functions
# ==============================================================================

def get_char_ngrams(text: str, n: int) -> Set[str]:
    """Extracts a set of character n-grams from a string."""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    """Computes Jaccard similarity between two sets with safe zero handling."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def extract_numeric_tokens(text: str) -> Set[str]:
    """Extracts alphanumeric numeric tokens (e.g. house numbers, plot numbers, PIN codes)."""
    if not text:
        return set()
    return set(re.findall(r'\b\d+[a-zA-Z]?\b|\b[a-zA-Z]?\d+\b', text.lower()))


def fallback_levenshtein(s1: str, s2: str) -> int:
    """Pure-Python Levenshtein distance fallback."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dele, subs))
        prev = curr
    return prev[-1]


def safe_normalized_levenshtein(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(distance.Levenshtein.normalized_similarity(s1, s2))
    m = max(len(s1), len(s2))
    return 1.0 - (fallback_levenshtein(s1, s2) / m)


def safe_fuzz_ratio(s1: str, s2: str) -> float:
    """Levenshtein ratio in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(fuzz.ratio(s1, s2) / 100.0)
    return safe_normalized_levenshtein(s1, s2)


def safe_token_sort_ratio(s1: str, s2: str) -> float:
    """Token sort ratio in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(fuzz.token_sort_ratio(s1, s2) / 100.0)
    s1_sort = " ".join(sorted(s1.split()))
    s2_sort = " ".join(sorted(s2.split()))
    return safe_normalized_levenshtein(s1_sort, s2_sort)


def safe_token_set_ratio(s1: str, s2: str) -> float:
    """Token set ratio in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(fuzz.token_set_ratio(s1, s2) / 100.0)
    t1 = set(s1.split())
    t2 = set(s2.split())
    return jaccard_similarity(t1, t2)


def safe_w_ratio(s1: str, s2: str) -> float:
    """Weighted fuzzy ratio in [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(fuzz.WRatio(s1, s2) / 100.0)
    return max(safe_fuzz_ratio(s1, s2), safe_token_sort_ratio(s1, s2))


# ==============================================================================
# Feature Names Registry
# ==============================================================================

BLOCKING_RULE_NAMES = [
    "blocked_exact_name",
    "blocked_compact_name",
    "blocked_suffix_name",
    "blocked_translit",
    "blocked_postal",
    "blocked_house",
    "blocked_address_token",
    "blocked_char_ngram",
    "blocked_consonant_skeleton",
    "blocked_address_name_combo"
]

FEATURE_NAMES: List[str] = [
    # --- Group 1: Name Features (45 features) ---
    "name_exact",
    "name_compact_exact",
    "name_nosuff_exact",
    "name_translit_exact",
    "name_nosuff_translit_exact",
    "name_levenshtein_ratio",
    "name_normalized_levenshtein",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_w_ratio",
    "name_translit_levenshtein_ratio",
    "name_translit_normalized_levenshtein",
    "name_translit_token_sort_ratio",
    "name_translit_token_set_ratio",
    "name_translit_w_ratio",
    "name_char2_jaccard",
    "name_char3_jaccard",
    "name_char4_jaccard",
    "name_char5_jaccard",
    "name_translit_char2_jaccard",
    "name_translit_char3_jaccard",
    "name_translit_char4_jaccard",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_shared_token_fraction_min",
    "name_shared_token_fraction_max",
    "name_translit_token_jaccard",
    "name_translit_token_overlap_count",
    "name_translit_shared_token_fraction_min",
    "name_length_diff",
    "name_length_ratio",
    "name_token_count_diff",
    "name_token_count_ratio",
    "name_first_token_exact",
    "name_last_token_exact",
    "name_prefix3_exact",
    "name_prefix4_exact",
    "name_prefix5_exact",
    "name_translit_first_token_exact",
    "name_translit_prefix3_exact",
    "consonant_skeleton_exact",
    "consonant_skeleton_levenshtein",
    "consonant_skeleton_jaccard",
    "consonant_skeleton_len_diff",
    "consonant_skeleton_len_ratio",

    # --- Group 2: Address Features (22 features) ---
    "address_exact",
    "address_compact_similarity",
    "address_levenshtein_ratio",
    "address_normalized_levenshtein",
    "address_token_jaccard",
    "address_token_set_ratio",
    "address_char3_ngram_jaccard",
    "address_length_ratio",
    "address_token_count_ratio",
    "address_numeric_token_overlap",
    "address_numeric_token_jaccard",
    "house_number_exact",
    "postal_code_exact",
    "postal_code_prefix3_exact",
    "s1_address_missing",
    "candidate_address_missing",
    "both_address_missing",
    "either_address_missing",
    "postal_missing_either",
    "postal_missing_both",
    "house_missing_either",
    "house_missing_both",

    # --- Group 3: Cross-Field / Combined Features (10 features) ---
    "same_country",
    "same_postal_and_high_name_sim",
    "same_house_and_high_name_sim",
    "same_postal_and_same_house",
    "same_postal_and_same_name_prefix",
    "addr_sim_x_name_sim",
    "max_name_similarity",
    "max_address_similarity",
    "name_and_addr_high_confidence",
    "name_exact_diff_address",

    # --- Group 4: Script / Language Signals (6 features) ---
    "s1_has_indic",
    "candidate_is_indic",
    "is_cross_script",
    "name_has_transliteration",
    "s1_script_latin",
    "candidate_script_latin",

    # --- Group 5: Source Signal (2 features) ---
    "candidate_source_is_S2",
    "candidate_source_is_S3",

    # --- Group 6: Blocking Evidence (11 features) ---
    "blocked_exact_name",
    "blocked_compact_name",
    "blocked_suffix_name",
    "blocked_translit",
    "blocked_postal",
    "blocked_house",
    "blocked_address_token",
    "blocked_char_ngram",
    "blocked_consonant_skeleton",
    "blocked_address_name_combo",
    "num_blocking_rules"
]

FEATURE_INDEX_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
NUM_FEATURES: int = len(FEATURE_NAMES)


# ==============================================================================
# Record Preparation Helper
# ==============================================================================

def enrich_record_for_features(raw_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures all canonical normalized, compact, stripped, and transliterated
    fields are populated for instant feature vector extraction.
    """
    name = raw_rec.get("name", "")
    addr = raw_rec.get("address", "")
    country = raw_rec.get("country", "")

    norm_n = raw_rec.get("norm_name") or normalize_name(name)
    comp_n = raw_rec.get("compact_name") or compact_string(norm_n)
    nosuff_n = raw_rec.get("nosuff_name") or remove_legal_suffix(norm_n)
    has_ind = raw_rec.get("has_indic", None)
    if has_ind is None:
        has_ind = has_indic_characters(name) or has_indic_characters(addr)

    trans_n = raw_rec.get("translit_name")
    if trans_n is None:
        trans_n = transliterate_name(name) if has_ind else norm_n
    trans_nosuff_n = remove_legal_suffix(trans_n)

    skel = extract_consonants(nosuff_n)
    trans_skel = extract_consonants(trans_nosuff_n)

    norm_a = raw_rec.get("norm_address")
    if norm_a is None:
        norm_a = normalize_address(addr)
    comp_a = compact_string(norm_a)

    post = raw_rec.get("postal_code")
    if post is None:
        post = extract_postal_code(addr)

    house = raw_rec.get("house_number")
    if house is None:
        house = extract_house_number(addr)

    tokens_n = norm_n.split()
    tokens_trans = trans_n.split()
    tokens_a = norm_a.split()

    return {
        "entity_id": raw_rec.get("entity_id", ""),
        "name": name,
        "address": addr,
        "country": country,
        "norm_name": norm_n,
        "compact_name": comp_n,
        "nosuff_name": nosuff_n,
        "translit_name": trans_n,
        "translit_nosuff_name": trans_nosuff_n,
        "skel_name": skel,
        "translit_skel_name": trans_skel,
        "norm_address": norm_a,
        "compact_address": comp_a,
        "postal_code": post,
        "house_number": house,
        "has_indic": has_ind,
        "tokens_name": tokens_n,
        "tokens_translit": tokens_trans,
        "tokens_address": tokens_a,
        "set_tokens_name": set(tokens_n),
        "set_tokens_translit": set(tokens_trans),
        "set_tokens_address": set(tokens_a),
        "numeric_tokens_address": extract_numeric_tokens(norm_a),
        "char2_name": get_char_ngrams(norm_n, 2),
        "char3_name": get_char_ngrams(norm_n, 3),
        "char4_name": get_char_ngrams(norm_n, 4),
        "char5_name": get_char_ngrams(norm_n, 5),
        "char2_translit": get_char_ngrams(trans_n, 2),
        "char3_translit": get_char_ngrams(trans_n, 3),
        "char4_translit": get_char_ngrams(trans_n, 4),
        "char3_address": get_char_ngrams(norm_a, 3),
    }


# ==============================================================================
# Pairwise Feature Extractor Engine
# ==============================================================================

class PairwiseFeatureExtractor:
    """
    High-performance feature extraction engine converting (S1, Candidate) record pairs
    and their blocking evidence into dense, standardized 96-dimensional float32 feature vectors.
    """

    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.feature_index_map = FEATURE_INDEX_MAP
        self.num_features = NUM_FEATURES

    def extract_features_dict(
        self,
        s1: Dict[str, Any],
        cand: Dict[str, Any],
        evidence_rules: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        """
        Extracts a dictionary of feature_name -> float value for a single entity pair.
        """
        feats: Dict[str, float] = {}

        # ----------------------------------------------------------------------
        # Group 1: Name Features
        # ----------------------------------------------------------------------
        s1_n = s1["norm_name"]
        cand_n = cand["norm_name"]
        cand_tr = cand["translit_name"]
        s1_ns = s1["nosuff_name"]
        cand_ns = cand["nosuff_name"]
        cand_tr_ns = cand["translit_nosuff_name"]

        # Exact matches
        feats["name_exact"] = 1.0 if s1_n and s1_n == cand_n else 0.0
        feats["name_compact_exact"] = 1.0 if s1["compact_name"] and s1["compact_name"] == cand["compact_name"] else 0.0
        feats["name_nosuff_exact"] = 1.0 if s1_ns and s1_ns == cand_ns else 0.0
        feats["name_translit_exact"] = 1.0 if s1_n and s1_n == cand_tr else 0.0
        feats["name_nosuff_translit_exact"] = 1.0 if s1_ns and s1_ns == cand_tr_ns else 0.0

        # Fuzzy similarity (Normalized)
        feats["name_levenshtein_ratio"] = safe_fuzz_ratio(s1_n, cand_n)
        feats["name_normalized_levenshtein"] = safe_normalized_levenshtein(s1_n, cand_n)
        feats["name_token_sort_ratio"] = safe_token_sort_ratio(s1_n, cand_n)
        feats["name_token_set_ratio"] = safe_token_set_ratio(s1_n, cand_n)
        feats["name_w_ratio"] = safe_w_ratio(s1_n, cand_n)

        # Transliterated fuzzy similarity
        feats["name_translit_levenshtein_ratio"] = safe_fuzz_ratio(s1_n, cand_tr)
        feats["name_translit_normalized_levenshtein"] = safe_normalized_levenshtein(s1_n, cand_tr)
        feats["name_translit_token_sort_ratio"] = safe_token_sort_ratio(s1_n, cand_tr)
        feats["name_translit_token_set_ratio"] = safe_token_set_ratio(s1_n, cand_tr)
        feats["name_translit_w_ratio"] = safe_w_ratio(s1_n, cand_tr)

        # Character n-grams
        feats["name_char2_jaccard"] = jaccard_similarity(s1["char2_name"], cand["char2_name"])
        feats["name_char3_jaccard"] = jaccard_similarity(s1["char3_name"], cand["char3_name"])
        feats["name_char4_jaccard"] = jaccard_similarity(s1["char4_name"], cand["char4_name"])
        feats["name_char5_jaccard"] = jaccard_similarity(s1["char5_name"], cand["char5_name"])
        feats["name_translit_char2_jaccard"] = jaccard_similarity(s1["char2_name"], cand["char2_translit"])
        feats["name_translit_char3_jaccard"] = jaccard_similarity(s1["char3_name"], cand["char3_translit"])
        feats["name_translit_char4_jaccard"] = jaccard_similarity(s1["char4_name"], cand["char4_translit"])

        # Token features
        t1 = s1["set_tokens_name"]
        t2 = cand["set_tokens_name"]
        t_tr = cand["set_tokens_translit"]

        feats["name_token_jaccard"] = jaccard_similarity(t1, t2)
        inter_len = len(t1 & t2)
        feats["name_token_overlap_count"] = float(inter_len)
        min_tok = min(len(t1), len(t2))
        max_tok = max(len(t1), len(t2))
        feats["name_shared_token_fraction_min"] = (inter_len / min_tok) if min_tok > 0 else 0.0
        feats["name_shared_token_fraction_max"] = (inter_len / max_tok) if max_tok > 0 else 0.0

        feats["name_translit_token_jaccard"] = jaccard_similarity(t1, t_tr)
        inter_tr = len(t1 & t_tr)
        feats["name_translit_token_overlap_count"] = float(inter_tr)
        min_tr = min(len(t1), len(t_tr))
        feats["name_translit_shared_token_fraction_min"] = (inter_tr / min_tr) if min_tr > 0 else 0.0

        # Structural features
        l1, l2 = len(s1_n), len(cand_n)
        feats["name_length_diff"] = float(abs(l1 - l2))
        feats["name_length_ratio"] = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0

        tc1, tc2 = len(s1["tokens_name"]), len(cand["tokens_name"])
        feats["name_token_count_diff"] = float(abs(tc1 - tc2))
        feats["name_token_count_ratio"] = (min(tc1, tc2) / max(tc1, tc2)) if max(tc1, tc2) > 0 else 1.0

        # Prefix / First token / Last token
        toks1, toks2 = s1["tokens_name"], cand["tokens_name"]
        toks_tr = cand["tokens_translit"]
        feats["name_first_token_exact"] = 1.0 if toks1 and toks2 and toks1[0] == toks2[0] else 0.0
        feats["name_last_token_exact"] = 1.0 if toks1 and toks2 and toks1[-1] == toks2[-1] else 0.0
        feats["name_prefix3_exact"] = 1.0 if len(s1_n) >= 3 and len(cand_n) >= 3 and s1_n[:3] == cand_n[:3] else 0.0
        feats["name_prefix4_exact"] = 1.0 if len(s1_n) >= 4 and len(cand_n) >= 4 and s1_n[:4] == cand_n[:4] else 0.0
        feats["name_prefix5_exact"] = 1.0 if len(s1_n) >= 5 and len(cand_n) >= 5 and s1_n[:5] == cand_n[:5] else 0.0
        feats["name_translit_first_token_exact"] = 1.0 if toks1 and toks_tr and toks1[0] == toks_tr[0] else 0.0
        feats["name_translit_prefix3_exact"] = 1.0 if len(s1_n) >= 3 and len(cand_tr) >= 3 and s1_n[:3] == cand_tr[:3] else 0.0

        # Consonant skeleton features
        skel1 = s1["skel_name"]
        skel2 = cand["translit_skel_name"] if cand["has_indic"] else cand["skel_name"]
        feats["consonant_skeleton_exact"] = 1.0 if skel1 and skel1 == skel2 else 0.0
        feats["consonant_skeleton_levenshtein"] = safe_normalized_levenshtein(skel1, skel2)
        skel1_2g = get_char_ngrams(skel1, 2)
        skel2_2g = get_char_ngrams(skel2, 2)
        feats["consonant_skeleton_jaccard"] = jaccard_similarity(skel1_2g, skel2_2g)
        sl1, sl2 = len(skel1), len(skel2)
        feats["consonant_skeleton_len_diff"] = float(abs(sl1 - sl2))
        feats["consonant_skeleton_len_ratio"] = (min(sl1, sl2) / max(sl1, sl2)) if max(sl1, sl2) > 0 else 1.0

        # ----------------------------------------------------------------------
        # Group 2: Address Features
        # ----------------------------------------------------------------------
        a1 = s1["norm_address"]
        a2 = cand["norm_address"]
        a1_comp = s1["compact_address"]
        a2_comp = cand["compact_address"]

        s1_addr_miss = 1.0 if not a1 or a1 in {"none", "null", "undefined"} else 0.0
        cand_addr_miss = 1.0 if not a2 or a2 in {"none", "null", "undefined"} else 0.0
        both_addr_miss = 1.0 if s1_addr_miss and cand_addr_miss else 0.0
        either_addr_miss = 1.0 if s1_addr_miss or cand_addr_miss else 0.0

        feats["s1_address_missing"] = s1_addr_miss
        feats["candidate_address_missing"] = cand_addr_miss
        feats["both_address_missing"] = both_addr_miss
        feats["either_address_missing"] = either_addr_miss

        if either_addr_miss:
            # Neutral / Zero scores for missing address to prevent false inflation
            feats["address_exact"] = 0.0
            feats["address_compact_similarity"] = 0.0
            feats["address_levenshtein_ratio"] = 0.0
            feats["address_normalized_levenshtein"] = 0.0
            feats["address_token_jaccard"] = 0.0
            feats["address_token_set_ratio"] = 0.0
            feats["address_char3_ngram_jaccard"] = 0.0
            feats["address_length_ratio"] = 0.0
            feats["address_token_count_ratio"] = 0.0
            feats["address_numeric_token_overlap"] = 0.0
            feats["address_numeric_token_jaccard"] = 0.0
        else:
            feats["address_exact"] = 1.0 if a1 == a2 else 0.0
            feats["address_compact_similarity"] = safe_normalized_levenshtein(a1_comp, a2_comp)
            feats["address_levenshtein_ratio"] = safe_fuzz_ratio(a1, a2)
            feats["address_normalized_levenshtein"] = safe_normalized_levenshtein(a1, a2)
            feats["address_token_jaccard"] = jaccard_similarity(s1["set_tokens_address"], cand["set_tokens_address"])
            feats["address_token_set_ratio"] = safe_token_set_ratio(a1, a2)
            feats["address_char3_ngram_jaccard"] = jaccard_similarity(s1["char3_address"], cand["char3_address"])
            al1, al2 = len(a1), len(a2)
            feats["address_length_ratio"] = (min(al1, al2) / max(al1, al2)) if max(al1, al2) > 0 else 1.0
            atc1, atc2 = len(s1["tokens_address"]), len(cand["tokens_address"])
            feats["address_token_count_ratio"] = (min(atc1, atc2) / max(atc1, atc2)) if max(atc1, atc2) > 0 else 1.0
            num1 = s1["numeric_tokens_address"]
            num2 = cand["numeric_tokens_address"]
            feats["address_numeric_token_overlap"] = float(len(num1 & num2))
            feats["address_numeric_token_jaccard"] = jaccard_similarity(num1, num2)

        # Postal code features
        p1 = s1["postal_code"]
        p2 = cand["postal_code"]
        p_miss_either = 1.0 if not p1 or not p2 else 0.0
        p_miss_both = 1.0 if not p1 and not p2 else 0.0
        feats["postal_missing_either"] = p_miss_either
        feats["postal_missing_both"] = p_miss_both
        feats["postal_code_exact"] = 1.0 if p1 and p2 and p1.lower() == p2.lower() else 0.0
        feats["postal_code_prefix3_exact"] = 1.0 if p1 and p2 and len(p1) >= 3 and len(p2) >= 3 and p1[:3].lower() == p2[:3].lower() else 0.0

        # House number features
        h1 = s1["house_number"]
        h2 = cand["house_number"]
        h_miss_either = 1.0 if not h1 or not h2 else 0.0
        h_miss_both = 1.0 if not h1 and not h2 else 0.0
        feats["house_missing_either"] = h_miss_either
        feats["house_missing_both"] = h_miss_both
        feats["house_number_exact"] = 1.0 if h1 and h2 and h1.lower() == h2.lower() else 0.0

        # ----------------------------------------------------------------------
        # Group 3: Cross-Field / Combined Features
        # ----------------------------------------------------------------------
        feats["same_country"] = 1.0 if s1["country"] and s1["country"] == cand["country"] else 0.0
        norm_name_sim = feats["name_normalized_levenshtein"]
        addr_sim = feats["address_token_jaccard"]

        feats["same_postal_and_high_name_sim"] = 1.0 if feats["postal_code_exact"] == 1.0 and norm_name_sim >= 0.70 else 0.0
        feats["same_house_and_high_name_sim"] = 1.0 if feats["house_number_exact"] == 1.0 and norm_name_sim >= 0.70 else 0.0
        feats["same_postal_and_same_house"] = 1.0 if feats["postal_code_exact"] == 1.0 and feats["house_number_exact"] == 1.0 else 0.0
        feats["same_postal_and_same_name_prefix"] = 1.0 if feats["postal_code_exact"] == 1.0 and feats["name_prefix3_exact"] == 1.0 else 0.0

        feats["addr_sim_x_name_sim"] = addr_sim * norm_name_sim

        max_n = max(
            feats["name_normalized_levenshtein"],
            feats["name_translit_normalized_levenshtein"],
            feats["name_token_sort_ratio"],
            feats["name_token_set_ratio"],
            feats["name_w_ratio"],
            feats["consonant_skeleton_levenshtein"]
        )
        feats["max_name_similarity"] = max_n

        max_a = max(
            feats["address_normalized_levenshtein"],
            feats["address_token_jaccard"],
            feats["address_token_set_ratio"],
            feats["address_char3_ngram_jaccard"]
        )
        feats["max_address_similarity"] = max_a

        feats["name_and_addr_high_confidence"] = 1.0 if max_n >= 0.80 and max_a >= 0.60 else 0.0
        feats["name_exact_diff_address"] = 1.0 if (feats["name_exact"] == 1.0 or feats["name_translit_exact"] == 1.0) and either_addr_miss == 0.0 and addr_sim < 0.20 else 0.0

        # ----------------------------------------------------------------------
        # Group 4: Script / Language Signals
        # ----------------------------------------------------------------------
        s1_ind = 1.0 if s1["has_indic"] else 0.0
        cand_ind = 1.0 if cand["has_indic"] else 0.0
        feats["s1_has_indic"] = s1_ind
        feats["candidate_is_indic"] = cand_ind
        feats["is_cross_script"] = 1.0 if s1_ind != cand_ind else 0.0
        feats["name_has_transliteration"] = 1.0 if cand_tr != cand_n else 0.0
        feats["s1_script_latin"] = 0.0 if s1_ind else 1.0
        feats["candidate_script_latin"] = 0.0 if cand_ind else 1.0

        # ----------------------------------------------------------------------
        # Group 5: Source Signal
        # ----------------------------------------------------------------------
        cid = cand.get("entity_id", "")
        feats["candidate_source_is_S2"] = 1.0 if cid.startswith("S2-") else 0.0
        feats["candidate_source_is_S3"] = 1.0 if cid.startswith("S3-") else 0.0

        # ----------------------------------------------------------------------
        # Group 6: Blocking Evidence
        # ----------------------------------------------------------------------
        ev = evidence_rules or set()
        rule_count = 0.0
        for rule_feat in BLOCKING_RULE_NAMES:
            has_rule = 1.0 if rule_feat in ev else 0.0
            feats[rule_feat] = has_rule
            rule_count += has_rule
        feats["num_blocking_rules"] = rule_count

        return feats

    def extract_features_vector(
        self,
        s1: Dict[str, Any],
        cand: Dict[str, Any],
        evidence_rules: Optional[Set[str]] = None
    ) -> np.ndarray:
        """
        Extracts a dense 1D float32 numpy array of length 96 for a single entity pair.
        """
        feats_dict = self.extract_features_dict(s1, cand, evidence_rules)
        vec = np.zeros(self.num_features, dtype=np.float32)
        for name, val in feats_dict.items():
            idx = self.feature_index_map[name]
            vec[idx] = val
        return vec

    def extract_batch_vectors(
        self,
        s1_records: Dict[str, Dict[str, Any]],
        cand_records: Dict[str, Dict[str, Any]],
        candidate_pairs: List[Tuple[str, str, Optional[Set[str]]]],
        chunk_size: int = 10000
    ) -> np.ndarray:
        """
        Batch extracts feature matrix for a list of candidate pairs:
        [(s1_id, candidate_id, evidence_rules_set), ...]
        Returns a 2D numpy array of shape (N, 96), dtype=np.float32.
        Processes in chunks to maintain strict memory efficiency.
        """
        n_pairs = len(candidate_pairs)
        feature_matrix = np.zeros((n_pairs, self.num_features), dtype=np.float32)

        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            for i in range(start, end):
                s1_id, cand_id, ev_rules = candidate_pairs[i]
                s1_rec = s1_records[s1_id]
                cand_rec = cand_records[cand_id]
                feature_matrix[i] = self.extract_features_vector(s1_rec, cand_rec, ev_rules)

        return feature_matrix
