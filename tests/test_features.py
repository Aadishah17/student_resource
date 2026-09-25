"""
ML Challenge 2026: Business Entity Resolution
Module: test_features.py

Comprehensive test suite verifying pairwise feature engineering:
- Exact matches
- Fuzzy matches
- Typos
- Legal suffix variation
- Word-order variation
- Missing address handling (zero inflation prevention, explicit indicators)
- Same postal code
- Same house number
- Cross-script pair handling
- Indic transliteration alignment
- Source attribution (S2 vs S3)
- Multi-rule blocking evidence tracking
- Vector sanity: exactly 96 dimensions, zero NaN, zero infinity, float32 dtype
"""

import os
import sys
import unittest
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))

from features import (
    PairwiseFeatureExtractor,
    enrich_record_for_features,
    FEATURE_NAMES,
    NUM_FEATURES
)


class TestPairwiseFeatures(unittest.TestCase):

    def setUp(self):
        self.extractor = PairwiseFeatureExtractor()

    def test_feature_registry_integrity(self):
        """Verify feature count and unique names."""
        self.assertEqual(len(FEATURE_NAMES), NUM_FEATURES)
        self.assertEqual(len(set(FEATURE_NAMES)), NUM_FEATURES)
        self.assertEqual(NUM_FEATURES, 96)

    def test_exact_matches(self):
        """Verify identical records produce perfect similarity scores."""
        r1 = enrich_record_for_features({
            "entity_id": "S1-001",
            "name": "Apex Logistics Limited",
            "address": "500 Industrial Pkwy, Dallas, TX 75201",
            "country": "US"
        })
        r2 = enrich_record_for_features({
            "entity_id": "S2-001",
            "name": "Apex Logistics Limited",
            "address": "500 Industrial Pkwy, Dallas, TX 75201",
            "country": "US"
        })
        feats = self.extractor.extract_features_dict(r1, r2)
        vec = self.extractor.extract_features_vector(r1, r2)

        self.assertEqual(feats["name_exact"], 1.0)
        self.assertEqual(feats["name_compact_exact"], 1.0)
        self.assertEqual(feats["name_nosuff_exact"], 1.0)
        self.assertAlmostEqual(feats["name_normalized_levenshtein"], 1.0)
        self.assertAlmostEqual(feats["name_token_jaccard"], 1.0)
        self.assertEqual(feats["address_exact"], 1.0)
        self.assertEqual(feats["postal_code_exact"], 1.0)
        self.assertEqual(feats["house_number_exact"], 1.0)
        self.assertEqual(feats["same_country"], 1.0)
        self.assertEqual(feats["s1_address_missing"], 0.0)
        self.assertEqual(feats["candidate_address_missing"], 0.0)

        # Vector sanity
        self.assertEqual(vec.shape, (96,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(np.isnan(vec).sum(), 0)
        self.assertEqual(np.isinf(vec).sum(), 0)

    def test_fuzzy_matches(self):
        """Verify fuzzy matches score high on Levenshtein and token metrics."""
        r1 = enrich_record_for_features({
            "entity_id": "S1-002",
            "name": "General Motors Corporation",
            "address": "300 Renaissance Center, Detroit, MI 48243",
            "country": "US"
        })
        r2 = enrich_record_for_features({
            "entity_id": "S2-002",
            "name": "General Motors Company",
            "address": "300 Renaissance Ctr, Detroit 48243",
            "country": "US"
        })
        feats = self.extractor.extract_features_dict(r1, r2)

        self.assertEqual(feats["name_exact"], 0.0)
        self.assertEqual(feats["name_nosuff_exact"], 1.0)
        self.assertGreater(feats["name_normalized_levenshtein"], 0.70)
        self.assertGreater(feats["name_token_sort_ratio"], 0.75)
        self.assertGreater(feats["address_token_jaccard"], 0.40)
        self.assertEqual(feats["postal_code_exact"], 1.0)

    def test_typos(self):
        """Verify single-character spelling typos are captured by fuzzy features."""
        r1 = enrich_record_for_features({
            "entity_id": "S1-003",
            "name": "Microsoft Corporation",
            "address": "1 Microsoft Way, Redmond, WA 98052",
            "country": "US"
        })
        r2 = enrich_record_for_features({
            "entity_id": "S3-003",
            "name": "Microsofft Corporation",
            "address": "1 Microsoft Way, Redmond, WA 98052",
            "country": "US"
        })
        feats = self.extractor.extract_features_dict(r1, r2)

        self.assertEqual(feats["name_exact"], 0.0)
        self.assertGreater(feats["name_levenshtein_ratio"], 0.90)
        self.assertGreater(feats["name_char3_jaccard"], 0.80)
        self.assertEqual(feats["address_exact"], 1.0)

    def test_legal_suffix_variation(self):
        """Verify legal suffix stripping bridges entity types."""
        r1 = enrich_record_for_features({
            "entity_id": "S1-004",
            "name": "Horizon Health Technologies Private Limited",
            "address": "Plot 14, Sector 5, Gurgaon 122001",
            "country": "India"
        })
        r2 = enrich_record_for_features({
            "entity_id": "S2-004",
            "name": "Horizon Health Technologies LLC",
            "address": "Plot 14, Sector 5, Gurgaon 122001",
            "country": "India"
        })
        feats = self.extractor.extract_features_dict(r1, r2)

        self.assertEqual(feats["name_exact"], 0.0)
        self.assertEqual(feats["name_nosuff_exact"], 1.0)
        self.assertEqual(feats["consonant_skeleton_exact"], 1.0)

    def test_word_order_variation(self):
        """Verify token sort ratio captures word re-ordering."""
        r1 = enrich_record_for_features({
            "entity_id": "S1-005",
            "name": "Alpha Delta Logistics",
            "address": "100 Main St, Chicago, IL 60601",
            "country": "US"
        })
        r2 = enrich_record_for_features({
            "entity_id": "S2-005",
            "name": "Delta Alpha Logistics",
            "address": "100 Main St, Chicago, IL 60601",
            "country": "US"
        })
        feats = self.extractor.extract_features_dict(r1, r2)

        self.assertEqual(feats["name_exact"], 0.0)
        self.assertAlmostEqual(feats["name_token_sort_ratio"], 1.0)
        self.assertAlmostEqual(feats["name_token_jaccard"], 1.0)

    def test_missing_address_handling(self):
        """Verify missing address produces explicit indicator and neutral similarities."""
        r1 = enrich_record_for_features({
            "entity_id": "S1-006",
            "name": "United Trading Corp",
            "address": "45 Park Avenue, New York, NY 10016",
            "country": "US"
        })
        r2 = enrich_record_for_features({
            "entity_id": "S3-006",
            "name": "United Trading Corp",
            "address": "",
            "country": "US"
        })
        feats = self.extractor.extract_features_dict(r1, r2)

        self.assertEqual(feats["s1_address_missing"], 0.0)
        self.assertEqual(feats["candidate_address_missing"], 1.0)
        self.assertEqual(feats["either_address_missing"], 1.0)
        self.assertEqual(feats["both_address_missing"], 0.0)
        self.assertEqual(feats["address_exact"], 0.0)
        self.assertEqual(feats["address_normalized_levenshtein"], 0.0)
        self.assertEqual(feats["address_token_jaccard"], 0.0)
        self.assertEqual(feats["postal_missing_either"], 1.0)
        self.assertEqual(feats["house_missing_either"], 1.0)

    def test_same_postal_and_house_number(self):
        """Verify postal and house number features."""
        r1 = enrich_record_for_features({
            "entity_id": "S1-007",
            "name": "Metro Retailers",
            "address": "Bungalow No 12, MG Road, Pune 411001",
            "country": "India"
        })
        r2 = enrich_record_for_features({
            "entity_id": "S2-007",
            "name": "Metro Retail Store",
            "address": "Plot 12, Camp, Pune 411001",
            "country": "India"
        })
        feats = self.extractor.extract_features_dict(r1, r2)

        self.assertEqual(feats["postal_code_exact"], 1.0)
        self.assertEqual(feats["postal_code_prefix3_exact"], 1.0)
        self.assertEqual(feats["house_number_exact"], 1.0)
        self.assertEqual(feats["same_postal_and_same_house"], 1.0)

    def test_cross_script_and_indic_transliteration(self):
        """Verify cross-script detection and transliterated feature alignment."""
        r1 = enrich_record_for_features({
            "entity_id": "S1-008",
            "name": "International Systems Private Limited",
            "address": "Flat 304, Bhagya Nagar, Hyderabad, Telangana 500037",
            "country": "India"
        })
        r2 = enrich_record_for_features({
            "entity_id": "S2-008",
            "name": "ఇంటర్నేషనల్ సిస్టమ్స్ ప్రైవేట్ లిమిటెడ్",
            "address": "304, BHAGYA NAGAR, Telangana 500037",
            "country": "India"
        })
        feats = self.extractor.extract_features_dict(r1, r2)

        self.assertEqual(feats["s1_has_indic"], 0.0)
        self.assertEqual(feats["candidate_is_indic"], 1.0)
        self.assertEqual(feats["is_cross_script"], 1.0)
        self.assertEqual(feats["name_has_transliteration"], 1.0)
        self.assertEqual(feats["s1_script_latin"], 1.0)
        self.assertEqual(feats["candidate_script_latin"], 0.0)
        self.assertGreater(feats["name_translit_normalized_levenshtein"], 0.65)
        self.assertGreater(feats["name_translit_char3_jaccard"], 0.20)
        self.assertGreater(feats["consonant_skeleton_levenshtein"], 0.70)
        self.assertEqual(feats["postal_code_exact"], 1.0)

    def test_source_signal(self):
        """Verify S2 vs S3 binary flags."""
        r1 = enrich_record_for_features({"entity_id": "S1-009", "name": "Test", "address": "123 Main", "country": "US"})
        r2_s2 = enrich_record_for_features({"entity_id": "S2-009", "name": "Test", "address": "123 Main", "country": "US"})
        r2_s3 = enrich_record_for_features({"entity_id": "S3-009", "name": "Test", "address": "123 Main", "country": "US"})

        feats_s2 = self.extractor.extract_features_dict(r1, r2_s2)
        feats_s3 = self.extractor.extract_features_dict(r1, r2_s3)

        self.assertEqual(feats_s2["candidate_source_is_S2"], 1.0)
        self.assertEqual(feats_s2["candidate_source_is_S3"], 0.0)
        self.assertEqual(feats_s3["candidate_source_is_S2"], 0.0)
        self.assertEqual(feats_s3["candidate_source_is_S3"], 1.0)

    def test_blocking_evidence_multiple_rules(self):
        """Verify blocking evidence indicators and rule count."""
        r1 = enrich_record_for_features({"entity_id": "S1-010", "name": "Alpha Corp", "address": "100 St", "country": "US"})
        r2 = enrich_record_for_features({"entity_id": "S2-010", "name": "Alpha Corp", "address": "100 St", "country": "US"})

        rules = {"blocked_exact_name", "blocked_postal", "blocked_consonant_skeleton"}
        feats = self.extractor.extract_features_dict(r1, r2, evidence_rules=rules)

        self.assertEqual(feats["blocked_exact_name"], 1.0)
        self.assertEqual(feats["blocked_postal"], 1.0)
        self.assertEqual(feats["blocked_consonant_skeleton"], 1.0)
        self.assertEqual(feats["blocked_house"], 0.0)
        self.assertEqual(feats["blocked_char_ngram"], 0.0)
        self.assertEqual(feats["num_blocking_rules"], 3.0)

    def test_batch_vector_extraction(self):
        """Verify batch vector extraction produces expected 2D numpy array without NaNs."""
        r1 = enrich_record_for_features({"entity_id": "S1-A", "name": "Apex Inc", "address": "100 St", "country": "US"})
        r2 = enrich_record_for_features({"entity_id": "S2-A", "name": "Apex Inc", "address": "100 St", "country": "US"})
        r3 = enrich_record_for_features({"entity_id": "S3-B", "name": "Beta LLC", "address": "200 Ave", "country": "US"})

        s1_records = {"S1-A": r1}
        cand_records = {"S2-A": r2, "S3-B": r3}
        pairs = [
            ("S1-A", "S2-A", {"blocked_exact_name"}),
            ("S1-A", "S3-B", {"blocked_address_token", "blocked_char_ngram"})
        ]

        mat = self.extractor.extract_batch_vectors(s1_records, cand_records, pairs)
        self.assertEqual(mat.shape, (2, 96))
        self.assertEqual(mat.dtype, np.float32)
        self.assertEqual(np.isnan(mat).sum(), 0)
        self.assertEqual(np.isinf(mat).sum(), 0)
        self.assertEqual(mat[0, FEATURE_NAMES.index("blocked_exact_name")], 1.0)
        self.assertEqual(mat[0, FEATURE_NAMES.index("num_blocking_rules")], 1.0)
        self.assertEqual(mat[1, FEATURE_NAMES.index("num_blocking_rules")], 2.0)


if __name__ == "__main__":
    unittest.main()
