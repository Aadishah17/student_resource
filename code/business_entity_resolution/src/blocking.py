"""
ML Challenge 2026: Business Entity Resolution
Module: blocking.py

Implements high-recall, memory-efficient candidate generation (blocking)
to reduce the enormous S1 x (S2 + S3) Cartesian space (~17-23 Trillion pairs)
into a compact, high-quality candidate set for subsequent ML scoring.

Key Design Principles:
1. Country-Aware Partitioning: Hard isolation across countries (open-set string labels).
2. Multi-Granular Name Inverted Indexes: Exact normalized, compact, legal-suffix-stripped,
   and Indic transliterated representations.
3. Discriminative Combined Keys: (postal + name prefix), (postal + house), (house + name prefix),
   and (postal + first token) preventing combinatorial explosion.
4. Character N-Gram Nearest-Neighbor Retrieval: Sparse TF-IDF (analyzer='char_wb')
   with chunked cosine dot product for fuzzy typo & variation coverage.
5. Invariant Address Anchors: Extracting discriminating street/locality/landmark tokens
   and building/plot numbers that bridge cross-script pairs.
6. Candidate Safety & Budget Controls: Bucket size caps (MAX_BUCKET_SIZE) to suppress
   pathological mega-buckets while allowing multi-candidate true matches.
7. 15.5 GB RAM Compliance: Chunked streaming, sparse CSR matrices, compact Python sets.
"""

import os
import sys
import time
import csv
from typing import Dict, List, Set, Tuple, Generator, Optional, Any
from collections import defaultdict, Counter
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# Import preprocessing helpers
try:
    from .preprocessing import (
        normalize_name,
        remove_legal_suffix,
        compact_string,
        transliterate_name,
        normalize_address,
        extract_postal_code,
        extract_house_number,
        has_indic_characters
    )
except ImportError:
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

# Common generic address terms to exclude from rare address token index
ADDR_STOPWORDS = {
    'street', 'st', 'road', 'rd', 'avenue', 'ave', 'blvd', 'boulevard', 'lane', 'ln',
    'drive', 'dr', 'court', 'ct', 'highway', 'hwy', 'parkway', 'pkwy', 'floor', 'fl',
    'suite', 'ste', 'unit', 'building', 'bldg', 'near', 'opp', 'opposite', 'beside',
    'behind', 'sector', 'sec', 'industrial', 'area', 'nagar', 'city', 'state', 'district',
    'north', 'south', 'east', 'west', 'null', 'undefined', 'rue', 'allee', 'france', 'usa', 'india',
    'floor', 'ground', 'first', 'second', 'third', 'fourth', 'number', 'main', 'bis', 'ter',
    'place', 'square', 'chemin', 'route', 'impasse'
}

# Maximum bucket size allowed for any single inverted index key to suppress degenerate tokens
DEFAULT_MAX_BUCKET_SIZE = 500
DEFAULT_MAX_HOUSE_BUCKET = 50
DEFAULT_MAX_ADDR_TOKEN_FREQ = 80


def extract_consonants(text: str) -> str:
    """Extracts only ASCII consonants from a string (stripping vowels a, e, i, o, u)."""
    vowels = set("aeiou")
    return "".join(c for c in text.lower() if c.isalpha() and c not in vowels)


# ==============================================================================
# 1. Inverted Index Blocker
# ==============================================================================

class InvertedIndexBlocker:
    """
    Manages country-partitioned inverted indexes across multiple lexical
    and address keys for business entity records.
    """

    def __init__(self, max_bucket_size: int = DEFAULT_MAX_BUCKET_SIZE):
        self.max_bucket_size = max_bucket_size
        # Structure: indexes[country][rule_name][key] -> list of entity_ids
        self.indexes: Dict[str, Dict[str, Dict[Any, List[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.addr_token_freq: Dict[str, Counter] = defaultdict(Counter)

    def compute_address_token_frequencies(self, records: List[Dict[str, Any]]):
        """Pre-computes address token frequencies to identify discriminating landmark/street tokens."""
        for rec in records:
            country = rec.get("country", "")
            norm_addr = rec.get("norm_address", "")
            if norm_addr:
                toks = set(t for t in norm_addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS)
                for t in toks:
                    self.addr_token_freq[country][t] += 1

    def add_record(self, rec: Dict[str, Any]):
        """Indexes a single target record (Source 2 or Source 3)."""
        eid = rec["entity_id"]
        country = rec.get("country", "")
        norm = rec.get("norm_name", "")
        comp = rec.get("compact_name", "")
        nosuff = rec.get("nosuff_name", "")
        trans = rec.get("translit_name", "")
        has_indic = rec.get("has_indic", False)
        post = rec.get("postal_code", "")
        house = rec.get("house_number", "")
        norm_addr = rec.get("norm_address", "")

        c_idx = self.indexes[country]

        # 1. Exact normalized name
        if norm and len(norm) >= 3:
            c_idx["exact_norm"][norm].append(eid)

        # 2. Exact compact name
        if comp and len(comp) >= 3:
            c_idx["exact_compact"][comp].append(eid)

        # 3. Suffix-stripped name
        if nosuff and len(nosuff) >= 3:
            c_idx["nosuff"][nosuff].append(eid)
            comp_ns = compact_string(nosuff)
            if comp_ns != comp:
                c_idx["nosuff_compact"][comp_ns].append(eid)

        # 4. Transliterated representation (critical for Indic records)
        if has_indic and trans:
            c_idx["translit"][trans].append(eid)
            tr_comp = compact_string(trans)
            if tr_comp:
                c_idx["translit_compact"][tr_comp].append(eid)
            tr_nosuff = remove_legal_suffix(trans)
            if tr_nosuff:
                c_idx["translit_nosuff"][tr_nosuff].append(eid)
                c_idx["translit_compact"][compact_string(tr_nosuff)].append(eid)

        # 5. Combined postal code keys
        effective_name = trans if has_indic else norm
        if post:
            if len(effective_name) >= 3:
                c_idx["postal_prefix"][(post, effective_name[:3])].append(eid)
            tokens = effective_name.split()
            if tokens and len(tokens[0]) >= 3:
                c_idx["postal_token"][(post, tokens[0])].append(eid)
            if house:
                c_idx["postal_house"][(post, house.lower())].append(eid)

        # 6. Combined house number keys
        if house and len(house) >= 2:
            c_idx["house"][house.lower()].append(eid)
            if len(effective_name) >= 3:
                c_idx["house_prefix"][(house.lower(), effective_name[:3])].append(eid)

        # 7. Discriminating rare address tokens
        if norm_addr:
            toks = [t for t in norm_addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and self.addr_token_freq[country][t] <= DEFAULT_MAX_ADDR_TOKEN_FREQ]
            for t in toks[:3]:
                c_idx["addr_token"][t].append(eid)

        # 8. Consonant skeleton key (vowel-invariant phonetic bridge)
        eff_nosuff = remove_legal_suffix(effective_name)
        if eff_nosuff:
            skel = extract_consonants(eff_nosuff)
            if len(skel) >= 4:
                c_idx["consonant_skel"][skel[:5]].append(eid)

        # 9. Address token + name prefix combination
        if norm_addr and len(effective_name) >= 2:
            toks = [t for t in norm_addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and self.addr_token_freq[country][t] <= 100]
            for t in toks[:3]:
                c_idx["addr_name"][(t, effective_name[:2])].append(eid)

    def query_with_evidence(self, s1_rec: Dict[str, Any], active_rules: Set[str]) -> Dict[str, Set[str]]:
        """
        Queries the inverted indexes for a single Source 1 entity under the active rules,
        returning a mapping of candidate_id -> set of blocking rule names that retrieved it.
        """
        evidence: Dict[str, Set[str]] = defaultdict(set)
        country = s1_rec.get("country", "")
        if country not in self.indexes:
            return evidence

        c_idx = self.indexes[country]
        norm = s1_rec.get("norm_name", "")
        comp = s1_rec.get("compact_name", "")
        nosuff = s1_rec.get("nosuff_name", "")
        post = s1_rec.get("postal_code", "")
        house = s1_rec.get("house_number", "").lower() if s1_rec.get("house_number") else ""
        norm_addr = s1_rec.get("norm_address", "")

        # Exact normalized & compact name
        if "exact" in active_rules:
            b = c_idx.get("exact_norm", {}).get(norm, [])
            if 0 < len(b) <= self.max_bucket_size:
                for eid in b:
                    evidence[eid].add("blocked_exact_name")
            b = c_idx.get("exact_compact", {}).get(comp, [])
            if 0 < len(b) <= self.max_bucket_size:
                for eid in b:
                    evidence[eid].add("blocked_compact_name")
            if nosuff:
                b = c_idx.get("nosuff", {}).get(nosuff, [])
                if 0 < len(b) <= self.max_bucket_size:
                    for eid in b:
                        evidence[eid].add("blocked_suffix_name")
                b = c_idx.get("nosuff_compact", {}).get(compact_string(nosuff), [])
                if 0 < len(b) <= self.max_bucket_size:
                    for eid in b:
                        evidence[eid].add("blocked_suffix_name")

        # Transliteration matching (query Latin S1 against Indic transliteration buckets)
        if "translit" in active_rules:
            b = c_idx.get("translit", {}).get(norm, [])
            if 0 < len(b) <= self.max_bucket_size:
                for eid in b:
                    evidence[eid].add("blocked_translit")
            if nosuff:
                b = c_idx.get("translit_nosuff", {}).get(nosuff, [])
                if 0 < len(b) <= self.max_bucket_size:
                    for eid in b:
                        evidence[eid].add("blocked_translit")
                b = c_idx.get("translit_compact", {}).get(compact_string(nosuff), [])
                if 0 < len(b) <= self.max_bucket_size:
                    for eid in b:
                        evidence[eid].add("blocked_translit")

        # Postal code combined keys
        if "postal" in active_rules and post:
            if len(norm) >= 3:
                b = c_idx.get("postal_prefix", {}).get((post, norm[:3]), [])
                if 0 < len(b) <= self.max_bucket_size:
                    for eid in b:
                        evidence[eid].add("blocked_postal")
            tokens = norm.split()
            if tokens and len(tokens[0]) >= 3:
                b = c_idx.get("postal_token", {}).get((post, tokens[0]), [])
                if 0 < len(b) <= self.max_bucket_size:
                    for eid in b:
                        evidence[eid].add("blocked_postal")

        # House number combined keys
        if "house" in active_rules:
            if post and house:
                b = c_idx.get("postal_house", {}).get((post, house), [])
                if 0 < len(b) <= self.max_bucket_size:
                    for eid in b:
                        evidence[eid].add("blocked_house")
            if house:
                if len(norm) >= 3:
                    b = c_idx.get("house_prefix", {}).get((house, norm[:3]), [])
                    if 0 < len(b) <= self.max_bucket_size:
                        for eid in b:
                            evidence[eid].add("blocked_house")
                # Specific house numbers with tight bucket cap
                b = c_idx.get("house", {}).get(house, [])
                if 0 < len(b) <= DEFAULT_MAX_HOUSE_BUCKET:
                    for eid in b:
                        evidence[eid].add("blocked_house")

        # Discriminating address tokens
        if "address" in active_rules and norm_addr:
            toks = [t for t in norm_addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and self.addr_token_freq[country][t] <= DEFAULT_MAX_ADDR_TOKEN_FREQ]
            for t in toks[:3]:
                b = c_idx.get("addr_token", {}).get(t, [])
                if 0 < len(b) <= DEFAULT_MAX_ADDR_TOKEN_FREQ:
                    for eid in b:
                        evidence[eid].add("blocked_address_token")

        # Consonant skeleton keys (vowel-invariant phonetic bridge)
        if ("consonant_skel" in active_rules or "skel" in active_rules):
            s1_nosuff = remove_legal_suffix(norm)
            if s1_nosuff:
                skel = extract_consonants(s1_nosuff)
                if len(skel) >= 4:
                    b = c_idx.get("consonant_skel", {}).get(skel[:5], [])
                    if 0 < len(b) <= 200:
                        for eid in b:
                            evidence[eid].add("blocked_consonant_skeleton")

        # Address token + name prefix combination
        if "addr_name" in active_rules and norm_addr and len(norm) >= 2:
            toks = [t for t in norm_addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOPWORDS and self.addr_token_freq[country][t] <= 100]
            for t in toks[:3]:
                b = c_idx.get("addr_name", {}).get((t, norm[:2]), [])
                if 0 < len(b) <= 200:
                    for eid in b:
                        evidence[eid].add("blocked_address_name_combo")

        return evidence

    def query(self, s1_rec: Dict[str, Any], active_rules: Set[str]) -> Set[str]:
        """Queries the inverted indexes for a single Source 1 entity under the active rules."""
        return set(self.query_with_evidence(s1_rec, active_rules).keys())


# ==============================================================================
# 2. Character N-Gram Nearest-Neighbor Blocker
# ==============================================================================

class NgramBlocker:
    """
    Performs nearest-neighbor retrieval in character n-gram TF-IDF space
    partitioned by country and target source, capturing spelling variants, typos,
    and transliterated suffixes.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 5),
        analyzer: str = "char_wb",
        min_df: int = 2,
        max_df: float = 0.25,
        top_k: int = 10,
        min_similarity: float = 0.40
    ):
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.min_df = min_df
        self.max_df = max_df
        self.top_k = top_k
        self.min_similarity = min_similarity

        self.models: Dict[str, Dict[str, Any]] = {}

    def fit_target_records(self, country: str, target_records: List[Dict[str, Any]]):
        """Fits TF-IDF vectorizer and builds sparse matrix for target records in a country."""
        if not target_records:
            return

        c_eids = [r["entity_id"] for r in target_records]
        # Use transliterated name for Indic records so n-grams overlap with Latin S1 queries
        c_names = [
            r["translit_name"] if r.get("has_indic", False) else r["norm_name"]
            for r in target_records
        ]

        n_docs = len(c_names)
        actual_min_df = self.min_df if n_docs >= 10 else 1
        actual_max_df = self.max_df if n_docs >= 10 else 1.0

        vec = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=actual_min_df,
            max_df=actual_max_df,
            dtype=np.float32
        )
        t_mat = vec.fit_transform(c_names)

        self.models[country] = {
            "vectorizer": vec,
            "target_matrix": t_mat,
            "target_eids": c_eids
        }

    def query_batch(
        self,
        country: str,
        s1_records: List[Dict[str, Any]],
        chunk_size: int = 500
    ) -> Dict[str, Set[str]]:
        """Queries a batch of S1 records against the fitted target matrix in chunks."""
        results: Dict[str, Set[str]] = {r["entity_id"]: set() for r in s1_records}
        if country not in self.models or not s1_records:
            return results

        model = self.models[country]
        vec = model["vectorizer"]
        t_mat = model["target_matrix"]
        t_eids = model["target_eids"]

        s1_names = [r["norm_name"] for r in s1_records]
        s1_mat = vec.transform(s1_names)

        n_queries = s1_mat.shape[0]
        for start in range(0, n_queries, chunk_size):
            end = min(start + chunk_size, n_queries)
            s1_chunk = s1_mat[start:end]
            sims = s1_chunk.dot(t_mat.T)  # Sparse CSR dot product

            for r in range(sims.shape[0]):
                s1_id = s1_records[start + r]["entity_id"]
                row = sims.getrow(r)
                if row.nnz == 0:
                    continue

                data = row.data
                indices = row.indices
                if len(data) > self.top_k:
                    # Argpartition for top_k largest values
                    part = np.argpartition(data, -self.top_k)[-self.top_k:]
                    part = part[np.argsort(-data[part])]
                    valid = [t_eids[indices[idx]] for idx in part if data[idx] >= self.min_similarity]
                else:
                    valid = [t_eids[indices[idx]] for idx in range(len(data)) if data[idx] >= self.min_similarity]

                results[s1_id].update(valid)

        return results


# ==============================================================================
# 3. Candidate Generation Pipeline
# ==============================================================================

class CandidateGenerator:
    """
    End-to-end candidate generation engine orchestrating country partitioning,
    inverted index retrieval, n-gram nearest-neighbor retrieval, candidate union,
    deduplication, and safety validation.
    """

    def __init__(
        self,
        active_rules: Optional[Set[str]] = None,
        use_ngram: bool = True,
        ngram_top_k: int = 10,
        ngram_min_sim: float = 0.40,
        max_bucket_size: int = DEFAULT_MAX_BUCKET_SIZE
    ):
        if active_rules is None:
            self.active_rules = {"exact", "postal", "house", "translit", "address"}
        else:
            self.active_rules = active_rules

        self.use_ngram = use_ngram
        self.ngram_top_k = ngram_top_k
        self.ngram_min_sim = ngram_min_sim
        self.max_bucket_size = max_bucket_size

        self.index_blocker = InvertedIndexBlocker(max_bucket_size=self.max_bucket_size)
        self.ngram_blocker = NgramBlocker(
            ngram_range=(3, 5),
            analyzer="char_wb",
            top_k=self.ngram_top_k,
            min_similarity=self.ngram_min_sim
        )

    def prepare_record(self, raw_row: Tuple[str, str, str, str]) -> Dict[str, Any]:
        """Converts raw TSV tuple (entity_id, name, address, country) into preprocessed representation."""
        eid, name, addr, country = raw_row
        norm_n = normalize_name(name)
        comp_n = compact_string(norm_n)
        nosuff_n = remove_legal_suffix(norm_n)
        has_ind = has_indic_characters(name)
        trans_n = transliterate_name(name) if has_ind else norm_n
        post = extract_postal_code(addr)
        house = extract_house_number(addr)
        norm_a = normalize_address(addr)

        return {
            "entity_id": eid,
            "name": name,
            "address": addr,
            "country": country,
            "norm_name": norm_n,
            "compact_name": comp_n,
            "nosuff_name": nosuff_n,
            "translit_name": trans_n,
            "has_indic": has_ind,
            "postal_code": post,
            "house_number": house,
            "norm_address": norm_a,
        }

    def fit_target_records(self, target_records: List[Dict[str, Any]]):
        """Indexes target records across all inverted index rules and n-gram models."""
        self.index_blocker.compute_address_token_frequencies(target_records)

        # Index into inverted indexes
        for rec in target_records:
            self.index_blocker.add_record(rec)

        # Fit n-gram models by country
        if self.use_ngram:
            by_country = defaultdict(list)
            for rec in target_records:
                by_country[rec.get("country", "")].append(rec)
            for country, recs in by_country.items():
                self.ngram_blocker.fit_target_records(country, recs)

    def generate_candidates_with_evidence(
        self,
        s1_records: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Set[str]]]:
        """
        Generates candidate pairs and records the exact set of blocking rules
        that contributed each candidate for every S1 record.
        Returns: {s1_id: {candidate_id: set_of_rule_names}}
        """
        candidates_evidence: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        # 1. Inverted index retrieval with evidence
        for s1_rec in s1_records:
            eid = s1_rec["entity_id"]
            ev = self.index_blocker.query_with_evidence(s1_rec, self.active_rules)
            for cid, rules in ev.items():
                if not cid.startswith("S1-"):
                    candidates_evidence[eid][cid].update(rules)

        # 2. Character n-gram retrieval
        if self.use_ngram:
            by_country = defaultdict(list)
            for rec in s1_records:
                by_country[rec.get("country", "")].append(rec)

            for country, recs in by_country.items():
                ngram_cands = self.ngram_blocker.query_batch(country, recs)
                for eid, cset in ngram_cands.items():
                    for cid in cset:
                        if not cid.startswith("S1-"):
                            candidates_evidence[eid][cid].add("blocked_char_ngram")

        # Format output ensuring all queries are present
        res: Dict[str, Dict[str, Set[str]]] = {}
        for s1_rec in s1_records:
            eid = s1_rec["entity_id"]
            res[eid] = dict(candidates_evidence[eid])
        return res

    def generate_candidates(
        self,
        s1_records: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Generates deduplicated candidates for a list of Source 1 records.
        Strict candidate safety guarantees:
        - Never returns S1 candidates.
        - Never returns cross-country candidates.
        - Deduplicated output per S1 entity.
        """
        evidence_map = self.generate_candidates_with_evidence(s1_records)
        return {eid: sorted(cands.keys()) for eid, cands in evidence_map.items()}

    @staticmethod
    def export_candidate_pairs_tsv(
        candidates_map: Dict[str, List[str]],
        output_path: str
    ):
        """
        Writes candidates in the required tab-separated format:
        source1_entity_id\tcandidate_entity_ids
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in sorted(candidates_map.keys()):
                cand_list = candidates_map[s1_id]
                cands_str = ",".join(cand_list)
                f.write(f"{s1_id}\t{cands_str}\n")


def create_config_h_generator(
    ngram_top_k: int = 10,
    ngram_min_sim: float = 0.25,
    max_bucket_size: int = DEFAULT_MAX_BUCKET_SIZE
) -> CandidateGenerator:
    """
    Factory function creating a CandidateGenerator configured with Configuration H:
    - Base rules: exact, postal, house, translit, address
    - Consonant skeleton keys
    - Address token + name prefix combination keys
    - Adaptive character n-gram retrieval
    """
    return CandidateGenerator(
        active_rules={
            "exact", "postal", "house", "translit", "address",
            "consonant_skel", "addr_name"
        },
        use_ngram=True,
        ngram_top_k=ngram_top_k,
        ngram_min_sim=ngram_min_sim,
        max_bucket_size=max_bucket_size
    )


# ==============================================================================
# 4. Validation & Evaluation Helper
# ==============================================================================

def evaluate_blocking_results(
    candidates_map: Dict[str, List[str]],
    ground_truth_map: Dict[str, Set[str]],
    s1_metadata: Dict[str, Dict[str, Any]],
    target_metadata: Dict[str, Dict[str, Any]],
    config_name: str = "Candidate Blocking"
) -> Dict[str, Any]:
    """
    Computes candidate recall, candidate volume statistics (mean, median, p95, max),
    reduction ratio, and subgroup recalls (US, India, Latin-Latin, Cross-Script).
    """
    total_true_links = sum(len(m) for m in ground_truth_map.values())
    counts = [len(candidates_map.get(s1_id, [])) for s1_id in ground_truth_map]
    total_pairs = sum(counts)

    found_total = 0
    found_us, total_us = 0, 0
    found_ind, total_ind = 0, 0
    found_latin, total_latin = 0, 0
    found_cross, total_cross = 0, 0

    for s1_id, true_matches in ground_truth_map.items():
        cands_set = set(candidates_map.get(s1_id, []))
        country = s1_metadata.get(s1_id, {}).get("country", "")

        for m in true_matches:
            # Sub-group accounting
            if country == "US":
                total_us += 1
            else:
                total_ind += 1

            m_rec = target_metadata.get(m, {})
            is_cross = m_rec.get("has_indic", False)
            if is_cross:
                total_cross += 1
            else:
                total_latin += 1

            # Match verification
            if m in cands_set:
                found_total += 1
                if country == "US":
                    found_us += 1
                else:
                    found_ind += 1
                if is_cross:
                    found_cross += 1
                else:
                    found_latin += 1

    rec_tot = found_total / total_true_links if total_true_links > 0 else 0.0
    rec_us = found_us / total_us if total_us > 0 else 0.0
    rec_ind = found_ind / total_ind if total_ind > 0 else 0.0
    rec_latin = found_latin / total_latin if total_latin > 0 else 0.0
    rec_cross = found_cross / total_cross if total_cross > 0 else 0.0

    cartesian_space = len(ground_truth_map) * len(target_metadata)
    reduction_ratio = 1.0 - (total_pairs / cartesian_space) if cartesian_space > 0 else 1.0

    return {
        "config_name": config_name,
        "recall_total": rec_tot,
        "recall_us": rec_us,
        "recall_ind": rec_ind,
        "recall_latin": rec_latin,
        "recall_cross_script": rec_cross,
        "mean_candidates_per_s1": float(np.mean(counts)),
        "median_candidates_per_s1": float(np.median(counts)),
        "p95_candidates_per_s1": float(np.percentile(counts, 95)),
        "max_candidates_per_s1": int(np.max(counts)) if counts else 0,
        "total_candidate_pairs": total_pairs,
        "reduction_ratio": reduction_ratio
    }
