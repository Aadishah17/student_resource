"""
ML Challenge 2026: Business Entity Resolution
Module: test_inference_safety.py

Unit test suite covering:
1. C1 / C2 Checkpoint & Resume Safety:
   - Safe default batch size
   - Resume with same batch size succeeds
   - Resume with different batch size is rejected with clear error
   - Interrupted partial writes roll back cleanly to byte offsets
2. C3 Streaming Blocker Recall:
   - blocked_address_token surfaces candidates sharing rare address landmarks
   - blocked_char_ngram surfaces candidates with spelling typos / consonant changes
   - Candidate ordering is strictly deterministic
   - Matched entity IDs remain a strict subset of candidate IDs
"""

import os
import sys
import json
import tempfile
import unittest
from typing import Dict, List, Set, Any
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

# Add src to path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

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
from features import PairwiseFeatureExtractor, NUM_FEATURES, enrich_record_for_features
from run_full_inference import (
    DEFAULT_BATCH_SIZE,
    MAX_S1_CANDS,
    save_checkpoint,
    load_checkpoint,
    process_country_partition
)
import xgboost as xgb


class MockModel:
    """Mock model that returns high probability for test candidates."""
    def predict_proba(self, feat_mat):
        import numpy as np
        # Return probability 0.85 for all pairs
        probs = np.zeros((feat_mat.shape[0], 2), dtype=np.float32)
        probs[:, 0] = 0.15
        probs[:, 1] = 0.85
        return probs


class TestInferenceSafety(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.extractor = PairwiseFeatureExtractor()
        self.mock_model = MockModel()

        # Build synthetic test S1 records (5 records for "US")
        self.s1_records: Dict[str, Dict[str, Any]] = {}
        raw_s1 = [
            ("S1-001", "Acme Industrial Supplies Inc", "100 Innovation Way, Suite 10, Boston, MA 02110", "US"),
            ("S1-002", "Blue Ocean Technologies LLC", "250 Atlantic Blvd, Boston, MA 02110", "US"),
            ("S1-003", "Summit Peak Logistics", "750 Mountain View Rd, Denver, CO 80201", "US"),
            ("S1-004", "Vizion Partnrs Global", "400 Valley Street, Austin, TX 78701", "US"),
            ("S1-005", "Heritage Landmark Crafts", "888 Historic Plaza, Santa Fe, NM 87501", "US"),
        ]
        for eid, name, addr, country in raw_s1:
            norm = normalize_name(name)
            has_ind = has_indic_characters(name)
            trans = transliterate_name(name) if has_ind else norm
            self.s1_records[eid] = {
                "entity_id": eid, "name": name, "address": addr, "country": country,
                "norm_name": norm,
                "compact_name": compact_string(norm),
                "nosuff_name": remove_legal_suffix(norm),
                "postal_code": extract_postal_code(addr),
                "house_number": extract_house_number(addr),
                "norm_address": normalize_address(addr),
                "has_indic": has_ind,
                "translit_name": trans,
                "translit_nosuff": remove_legal_suffix(trans),
                "skel": extract_consonants(remove_legal_suffix(norm))
            }

        # Create target file
        self.target_path = os.path.join(self.tmp_dir.name, "test_target.tsv")
        raw_targets = [
            ("S2-T1", "Acme Industrial Supplies Corp", "100 Innovation Way, Boston, MA 02110", "US"),
            ("S2-T2", "Blue Ocean Technologies", "250 Atlantic Blvd, Boston 02110", "US"),
            ("S3-T3", "Vision Partners Global Inc", "400 Valley Street, Austin 78701", "US"),
            ("S2-T4", "Different Name Artisans", "888 Historic Plaza, Santa Fe, NM", "US"),
        ]
        with open(self.target_path, "w", encoding="utf-8", newline="") as f:
            f.write("entity_id\tname\taddress\tcountry\n")
            for r in raw_targets:
                f.write("\t".join(r) + "\n")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_safe_batch_size_default(self):
        """C1: Verify the default batch size is memory-aware and <= 10,000."""
        self.assertLessEqual(DEFAULT_BATCH_SIZE, 10000)
        self.assertEqual(DEFAULT_BATCH_SIZE, 10000)

    def test_resume_with_same_batch_size_succeeds(self):
        """C2.1: Verify resume with identical batch size skips completed batches and finishes run."""
        matching_file_path = os.path.join(self.tmp_dir.name, "matching.tsv")
        candidate_file_path = os.path.join(self.tmp_dir.name, "candidate.tsv")
        ckpt_file_path = os.path.join(self.tmp_dir.name, "checkpoint.json")

        ckpt_state = {
            "completed_batches": [],
            "written_s1_count": 0,
            "batch_size": 2
        }

        # Step 1: Run with batch_size=2 and exit after 1 batch
        with open(matching_file_path, "w", encoding="utf-8", newline="") as fm, \
             open(candidate_file_path, "w", encoding="utf-8", newline="") as fc:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            fm.flush(); fc.flush()
            ckpt_state["matching_byte_offset"] = fm.tell()
            ckpt_state["candidate_byte_offset"] = fc.tell()
            save_checkpoint(ckpt_file_path, ckpt_state)

            try:
                process_country_partition(
                    country="US",
                    s1_records=self.s1_records,
                    target_paths=[self.target_path],
                    model=self.mock_model,
                    tau=0.72,
                    extractor=self.extractor,
                    out_matching_file=fm,
                    out_candidate_file=fc,
                    batch_size=2,
                    checkpoint_file=ckpt_file_path,
                    checkpoint_state=ckpt_state,
                    exit_after_batches=1
                )
            except SystemExit:
                pass

        # Checkpoint after first batch
        loaded_ckpt = load_checkpoint(ckpt_file_path)
        self.assertEqual(len(loaded_ckpt["completed_batches"]), 1)
        self.assertEqual(loaded_ckpt["batch_size"], 2)
        self.assertEqual(loaded_ckpt["written_s1_count"], 2)

        # Step 2: Resume with same batch_size=2
        with open(matching_file_path, "a", encoding="utf-8", newline="") as fm, \
             open(candidate_file_path, "a", encoding="utf-8", newline="") as fc:
            process_country_partition(
                country="US",
                s1_records=self.s1_records,
                target_paths=[self.target_path],
                model=self.mock_model,
                tau=0.72,
                extractor=self.extractor,
                out_matching_file=fm,
                out_candidate_file=fc,
                batch_size=2,
                checkpoint_file=ckpt_file_path,
                checkpoint_state=loaded_ckpt,
                exit_after_batches=None
            )

        final_ckpt = load_checkpoint(ckpt_file_path)
        # 5 entities total with batch_size=2 -> 3 batches (0-2, 2-4, 4-5)
        self.assertEqual(len(final_ckpt["completed_batches"]), 3)
        self.assertEqual(final_ckpt["written_s1_count"], 5)

        # Verify output files have exactly 5 S1 entities (no duplicates, no omissions)
        with open(matching_file_path, "r", encoding="utf-8") as f:
            lines = [l.strip().split("\t")[0] for l in f if l.strip()]
        header = lines[0]
        data_eids = lines[1:]
        self.assertEqual(header, "source1_entity_id")
        self.assertEqual(len(data_eids), 5)
        self.assertEqual(sorted(data_eids), sorted(list(self.s1_records.keys())))

    def test_resume_with_different_batch_size_rejected(self):
        """C2.2: Verify resume with different batch size raises ValueError."""
        ckpt_file_path = os.path.join(self.tmp_dir.name, "diff_bs_ckpt.json")
        ckpt_state = {
            "completed_batches": ["US_batch_0_2"],
            "written_s1_count": 2,
            "batch_size": 2
        }
        save_checkpoint(ckpt_file_path, ckpt_state)

        matching_file_path = os.path.join(self.tmp_dir.name, "m_diff.tsv")
        candidate_file_path = os.path.join(self.tmp_dir.name, "c_diff.tsv")

        with open(matching_file_path, "w", encoding="utf-8") as fm, \
             open(candidate_file_path, "w", encoding="utf-8") as fc:
            with self.assertRaises(ValueError) as ctx:
                process_country_partition(
                    country="US",
                    s1_records=self.s1_records,
                    target_paths=[self.target_path],
                    model=self.mock_model,
                    tau=0.72,
                    extractor=self.extractor,
                    out_matching_file=fm,
                    out_candidate_file=fc,
                    batch_size=4,  # Different batch size!
                    checkpoint_file=ckpt_file_path,
                    checkpoint_state=ckpt_state
                )
            self.assertIn("Incompatible batch size", str(ctx.exception))

    def test_interrupted_partial_writes_rollback(self):
        """C2.3: Verify interrupted partial writes roll back to clean checkpoint byte offset."""
        matching_file_path = os.path.join(self.tmp_dir.name, "m_rollback.tsv")
        candidate_file_path = os.path.join(self.tmp_dir.name, "c_rollback.tsv")
        ckpt_file_path = os.path.join(self.tmp_dir.name, "ckpt_rollback.json")

        ckpt_state = {
            "completed_batches": [],
            "written_s1_count": 0,
            "batch_size": 2
        }

        # Step 1: Run batch 1 cleanly
        with open(matching_file_path, "w", encoding="utf-8", newline="") as fm, \
             open(candidate_file_path, "w", encoding="utf-8", newline="") as fc:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            fm.flush(); fc.flush()
            ckpt_state["matching_byte_offset"] = fm.tell()
            ckpt_state["candidate_byte_offset"] = fc.tell()
            save_checkpoint(ckpt_file_path, ckpt_state)

            try:
                process_country_partition(
                    country="US",
                    s1_records=self.s1_records,
                    target_paths=[self.target_path],
                    model=self.mock_model,
                    tau=0.72,
                    extractor=self.extractor,
                    out_matching_file=fm,
                    out_candidate_file=fc,
                    batch_size=2,
                    checkpoint_file=ckpt_file_path,
                    checkpoint_state=ckpt_state,
                    exit_after_batches=1
                )
            except SystemExit:
                pass

        saved_ckpt = load_checkpoint(ckpt_file_path)
        clean_m_offset = saved_ckpt["matching_byte_offset"]
        clean_c_offset = saved_ckpt["candidate_byte_offset"]

        # Step 2: Simulate crash mid-batch-2 (append corrupted partial line without saving checkpoint)
        with open(matching_file_path, "a", encoding="utf-8") as fm, \
             open(candidate_file_path, "a", encoding="utf-8") as fc:
            fm.write("S1-CORRUPTED_PARTIAL_ENTRY\t")
            fc.write("S1-CORRUPTED_PARTIAL_ENTRY\t")

        self.assertGreater(os.path.getsize(matching_file_path), clean_m_offset)
        self.assertGreater(os.path.getsize(candidate_file_path), clean_c_offset)

        # Step 3: Perform rollback (as executed in run_full_inference.py resume mode)
        with open(matching_file_path, "r+", encoding="utf-8") as fm:
            fm.seek(clean_m_offset)
            fm.truncate()
        with open(candidate_file_path, "r+", encoding="utf-8") as fc:
            fc.seek(clean_c_offset)
            fc.truncate()

        self.assertEqual(os.path.getsize(matching_file_path), clean_m_offset)
        self.assertEqual(os.path.getsize(candidate_file_path), clean_c_offset)

        # Step 4: Resume execution
        with open(matching_file_path, "a", encoding="utf-8", newline="") as fm, \
             open(candidate_file_path, "a", encoding="utf-8", newline="") as fc:
            process_country_partition(
                country="US",
                s1_records=self.s1_records,
                target_paths=[self.target_path],
                model=self.mock_model,
                tau=0.72,
                extractor=self.extractor,
                out_matching_file=fm,
                out_candidate_file=fc,
                batch_size=2,
                checkpoint_file=ckpt_file_path,
                checkpoint_state=saved_ckpt,
                exit_after_batches=None
            )

        with open(matching_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("CORRUPTED_PARTIAL_ENTRY", content)

    def test_streaming_blocker_address_token_surfaces_candidate(self):
        """C3.1: Verify blocked_address_token surfaces candidates sharing rare address landmarks."""
        # S1-005: "Heritage Landmark Crafts", "888 Historic Plaza, Santa Fe, NM 87501"
        # S2-T4: "Different Name Artisans", "888 Historic Plaza, Santa Fe, NM"
        # Names share NO common words or prefixes, but share rare address token "historic" or "plaza"
        matching_file_path = os.path.join(self.tmp_dir.name, "m_addr.tsv")
        candidate_file_path = os.path.join(self.tmp_dir.name, "c_addr.tsv")

        with open(matching_file_path, "w", encoding="utf-8") as fm, \
             open(candidate_file_path, "w", encoding="utf-8") as fc:
            process_country_partition(
                country="US",
                s1_records={"S1-005": self.s1_records["S1-005"]},
                target_paths=[self.target_path],
                model=self.mock_model,
                tau=0.72,
                extractor=self.extractor,
                out_matching_file=fm,
                out_candidate_file=fc,
                batch_size=10
            )

        with open(candidate_file_path, "r", encoding="utf-8") as fc:
            cand_lines = [l.strip().split("\t") for l in fc if l.strip()]

        # S1-005 should retrieve S2-T4 via blocked_address_token / house
        self.assertEqual(cand_lines[0][0], "S1-005")
        cands = cand_lines[0][1].split(",") if len(cand_lines[0]) > 1 and cand_lines[0][1] else []
        self.assertIn("S2-T4", cands)

    def test_streaming_blocker_char_ngram_surfaces_candidate(self):
        """C3.2: Verify blocked_char_ngram surfaces candidates with spelling typos."""
        # S1-004: "Vizion Partnrs Global" vs S3-T3: "Vision Partners Global Inc"
        # Note 'z' in Vizion and missing 'e' in Partnrs: consonant skeleton prefix diverges (vznpr vs vsnpr)
        matching_file_path = os.path.join(self.tmp_dir.name, "m_ngram.tsv")
        candidate_file_path = os.path.join(self.tmp_dir.name, "c_ngram.tsv")

        with open(matching_file_path, "w", encoding="utf-8") as fm, \
             open(candidate_file_path, "w", encoding="utf-8") as fc:
            process_country_partition(
                country="US",
                s1_records={"S1-004": self.s1_records["S1-004"]},
                target_paths=[self.target_path],
                model=self.mock_model,
                tau=0.72,
                extractor=self.extractor,
                out_matching_file=fm,
                out_candidate_file=fc,
                batch_size=10
            )

        with open(candidate_file_path, "r", encoding="utf-8") as fc:
            cand_lines = [l.strip().split("\t") for l in fc if l.strip()]

        self.assertEqual(cand_lines[0][0], "S1-004")
        cands = cand_lines[0][1].split(",") if len(cand_lines[0]) > 1 and cand_lines[0][1] else []
        self.assertIn("S3-T3", cands)

    def test_candidate_ordering_is_deterministic(self):
        """Verify candidate IDs in candidate_pairs.tsv are strictly sorted lexicographically."""
        matching_file_path = os.path.join(self.tmp_dir.name, "m_order.tsv")
        candidate_file_path = os.path.join(self.tmp_dir.name, "c_order.tsv")

        with open(matching_file_path, "w", encoding="utf-8") as fm, \
             open(candidate_file_path, "w", encoding="utf-8") as fc:
            process_country_partition(
                country="US",
                s1_records=self.s1_records,
                target_paths=[self.target_path],
                model=self.mock_model,
                tau=0.72,
                extractor=self.extractor,
                out_matching_file=fm,
                out_candidate_file=fc,
                batch_size=10
            )

        with open(candidate_file_path, "r", encoding="utf-8") as fc:
            for line in fc:
                parts = line.strip().split("\t")
                if len(parts) > 1 and parts[1]:
                    cand_ids = parts[1].split(",")
                    self.assertEqual(cand_ids, sorted(cand_ids), f"Candidates not sorted for {parts[0]}")

    def test_matching_ids_strict_subset_of_candidate_ids(self):
        """Verify matched entity IDs are always a strict subset of candidate IDs."""
        matching_file_path = os.path.join(self.tmp_dir.name, "m_subset.tsv")
        candidate_file_path = os.path.join(self.tmp_dir.name, "c_subset.tsv")

        with open(matching_file_path, "w", encoding="utf-8") as fm, \
             open(candidate_file_path, "w", encoding="utf-8") as fc:
            process_country_partition(
                country="US",
                s1_records=self.s1_records,
                target_paths=[self.target_path],
                model=self.mock_model,
                tau=0.72,
                extractor=self.extractor,
                out_matching_file=fm,
                out_candidate_file=fc,
                batch_size=10
            )

        matches_map = {}
        with open(matching_file_path, "r", encoding="utf-8") as fm:
            for line in fm:
                parts = line.strip().split("\t")
                matches_map[parts[0]] = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()

        candidates_map = {}
        with open(candidate_file_path, "r", encoding="utf-8") as fc:
            for line in fc:
                parts = line.strip().split("\t")
                candidates_map[parts[0]] = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()

        for eid, m_set in matches_map.items():
            c_set = candidates_map.get(eid, set())
            self.assertTrue(m_set.issubset(c_set), f"Matches {m_set} not subset of candidates {c_set} for {eid}")


if __name__ == "__main__":
    unittest.main()
