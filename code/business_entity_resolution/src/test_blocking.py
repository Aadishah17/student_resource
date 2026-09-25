"""
ML Challenge 2026: Business Entity Resolution
Module: test_blocking.py

Unit test suite for blocking.py ensuring:
- Strict country partitioning (no cross-country leaks)
- Exact normalized and compact name retrieval
- Legal suffix stripped matching
- Indic transliteration cross-script matching
- Combined postal and house number matching
- Character n-gram fuzzy matching
- Candidate safety (no S1 IDs, valid TSV export)
"""

import os
import sys
import unittest
import tempfile

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from blocking import (
    InvertedIndexBlocker,
    NgramBlocker,
    CandidateGenerator,
    evaluate_blocking_results
)


class TestBlocking(unittest.TestCase):

    def setUp(self):
        self.generator = CandidateGenerator(
            active_rules={"exact", "postal", "house", "translit", "address"},
            use_ngram=True,
            ngram_top_k=5,
            ngram_min_sim=0.30
        )

        # Target records from S2 and S3
        self.raw_targets = [
            # US Targets
            ("S2-001", "Orelee's Barbershop Inc.", "1795 Westchester Drive, High Point, NC 27262", "US"),
            ("S3-002", "Vision Partners Corp", "1064 Newton Rd, Unit 11, Iowa City, IA", "US"),
            ("S2-003", "Apex Logistics Limited", "500 Industrial Pkwy, Dallas, TX 75201", "US"),

            # India Targets (Mixed Latin & Indic)
            ("S2-004", "ఇంటర్నేషనల్ సిస్టమ్స్ ప్రైవేట్ లిమిటెడ్", "304, 4TH FLOOR, BHAGYA NAGAR, Telangana 500037", "India"),
            ("S3-005", "लक्ष्मी डेवलपर्स प्राइवेट लिमिटेड", "H.NO #1338, NESARI, KOLHAPUR, Maharashtra 416001", "India"),
            ("S2-006", "Pioneer Tech Solutions Pvt Ltd", "RZ-142, Ground Floor, Vishnu Garden, New Delhi 110018", "India"),

            # France Target
            ("S2-007", "Thermal & Fils SASU", "20 Rue Parmentier, Dunkerque 59140", "France"),
        ]

        self.target_recs = [self.generator.prepare_record(r) for r in self.raw_targets]
        self.generator.fit_target_records(self.target_recs)

    def test_country_partitioning(self):
        """Verify strict isolation: US queries never return Indian or French candidates."""
        s1_us = [self.generator.prepare_record(("S1-US-1", "Apex Logistics Inc", "500 Industrial Parkway, Dallas 75201", "US"))]
        cands = self.generator.generate_candidates(s1_us)

        us_matches = cands["S1-US-1"]
        self.assertIn("S2-003", us_matches)
        # Verify no Indian or French records leaked
        for non_us_id in ["S2-004", "S3-005", "S2-006", "S2-007"]:
            self.assertNotIn(non_us_id, us_matches)

    def test_exact_and_suffix_stripped_blocking(self):
        """Verify normalized and suffix-stripped matching."""
        s1 = [self.generator.prepare_record(("S1-100", "Orelees Barbershop", "1795 Westchester Dr", "US"))]
        cands = self.generator.generate_candidates(s1)
        self.assertIn("S2-001", cands["S1-100"])

    def test_transliteration_cross_script_blocking(self):
        """Verify Indic transliterated records match English/Latin S1 queries."""
        # Query: English "International Systems" matching Telugu "ఇంటర్నేషనల్ సిస్టమ్స్"
        s1_telugu = [self.generator.prepare_record(("S1-IND-1", "International Systems Pvt Ltd", "Bhagya Nagar, Telangana", "India"))]
        cands = self.generator.generate_candidates(s1_telugu)
        self.assertIn("S2-004", cands["S1-IND-1"])

    def test_postal_and_house_number_blocking(self):
        """Verify combined postal code and house number retrieval."""
        s1_house = [self.generator.prepare_record(("S1-IND-2", "Lakshmi Dev Corp", "H.No 1338, Kolhapur 416001", "India"))]
        cands = self.generator.generate_candidates(s1_house)
        self.assertIn("S3-005", cands["S1-IND-2"])

    def test_ngram_fuzzy_retrieval(self):
        """Verify character n-gram retrieval catches minor spelling typos."""
        # Typo: "Vision Partnrs" instead of "Vision Partners"
        s1_typo = [self.generator.prepare_record(("S1-US-2", "Vision Partnrs", "Newton Rd", "US"))]
        cands = self.generator.generate_candidates(s1_typo)
        self.assertIn("S3-002", cands["S1-US-2"])

    def test_candidate_safety_no_s1_ids(self):
        """Verify candidate sets never contain S1 entity IDs."""
        s1 = [self.generator.prepare_record(("S1-TEST", "Thermal & Fils", "20 Rue Parmentier", "France"))]
        cands = self.generator.generate_candidates(s1)
        for cid in cands["S1-TEST"]:
            self.assertFalse(cid.startswith("S1-"), f"Found forbidden S1 ID {cid} in candidate list")
        self.assertIn("S2-007", cands["S1-TEST"])

    def test_tsv_export_format(self):
        """Verify TSV export conforms strictly to challenge specification."""
        cands = {
            "S1-001": ["S2-001", "S3-002"],
            "S1-002": []  # Singleton
        }
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".tsv") as tmp:
            tmp_path = tmp.name

        try:
            CandidateGenerator.export_candidate_pairs_tsv(cands, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\r\n") for line in f]

            self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
            self.assertEqual(lines[1], "S1-001\tS2-001,S3-002")
            self.assertEqual(lines[2], "S1-002\t")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
