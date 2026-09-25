"""
Unit and integration test script for preprocessing.py
Tests representative business entity records across US, India, and France.
"""

import os
import sys
import unittest
import pandas as pd

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

# Add the source directory to the import path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))
from preprocessing import (
    normalize_name,
    remove_legal_suffix,
    compact_string,
    tokenize_string,
    transliterate_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
    extract_house_number,
    preprocess_record,
    preprocess_dataframe,
)


class TestPreprocessing(unittest.TestCase):

    def test_name_normalization_and_legal_suffixes(self):
        """Test name normalization, & vs and, punctuation, and legal suffix removal."""
        cases = [
            ("Orelee's Barbershop Inc.", "orelees barbershop", "orelees barbershop"),
            ("B+ Retail & Co. LLC", "b retail", "b retail"),
            ("Whitley Inc Quasaredge", "whitley quasaredge", "whitley quasaredge"),
            ("International Systems Private Limited", "international systems", "international systems"),
            ("Thermal & Fils SASU", "thermal and fils", "thermal and fils"),
            ("SCI Ptit Àmicale", "ptit amicale", "ptit amicale"),
            ("Elephant Centre EURL", "elephant centre", "elephant centre"),
            ("Fractales Amis Groupe S.A.S", "fractales amis groupe", "fractales amis groupe"),
            ("M/s Haven Exim Co", "haven exim", "haven exim"),
            ("-- Holloway Peak Inc Seafood", "holloway peak seafood", "holloway peak seafood"),
            ("<< Team Ecole", "team ecole", "team ecole"),
        ]
        for raw_name, expected_core_substr, _ in cases:
            norm = normalize_name(raw_name)
            no_suffix = remove_legal_suffix(norm)
            self.assertTrue(len(norm) > 0, f"Failed on {raw_name}")
            self.assertIn(expected_core_substr, no_suffix, f"Expected '{expected_core_substr}' in '{no_suffix}' for raw '{raw_name}'")

    def test_address_normalization_abbreviations(self):
        """Test address normalization, abbreviations expansion, and noise stripping."""
        cases = [
            ("1795 Westchester Dr., High Point, NC", "drive"),
            ("105 ELM ST, MORGANTON, NC", "street"),
            ("8935 Georgetown Pike, Fairfax County, VA", "pike"),
            ("6207 OCEAN FRONT AVE, VIRGINIA BEACH CITY, VA", "avenue"),
            ("18 RUE JEN ZAY, Dunkerque, Nord", "rue"),
            ("NO. 5 ALLÉE DES HÊTRES, Pornic", "allee"),
            ("63 R. DE DIEPPE, LILLE, Hauts-de-France", "rue"),
            ("459 Carter Rd, null, Plymouth, Connecticut", "road"),
        ]
        for raw_addr, expected_term in cases:
            norm = normalize_address(raw_addr)
            self.assertNotIn("null", norm)
            self.assertIn(expected_term, norm, f"Expected '{expected_term}' in '{norm}' for raw '{raw_addr}'")

    def test_missing_address_handling(self):
        """Test safe handling of None, NaN, and empty string addresses."""
        for val in ["", None, float("nan")]:
            norm = normalize_address(val)
            comp = compact_string(norm)
            toks = tokenize_string(norm)
            postal = extract_postal_code(val)
            nums = extract_numeric_tokens(val)
            house = extract_house_number(val)

            self.assertEqual(norm, "")
            self.assertEqual(comp, "")
            self.assertEqual(toks, ())
            self.assertEqual(postal, "")
            self.assertEqual(nums, ())
            self.assertEqual(house, "")

    def test_indic_transliteration(self):
        """Test Indic-to-Latin phonetic transliteration across multiple scripts."""
        cases = [
            ("ఇంటర్నేషనల్ సిస్టమ్స్ ప్రైవేట్ లిమిటెడ్", ["intarneshanal", "sistams", "praivet", "limited"]),
            ("பிரைம் புராஜெக்ட்ஸ் பிரைவேட் லிமிடெட்", ["piraim", "puraajekts", "piraivet", "limitet"]),
            ("लक्ष्मी डेवलपर्स प्राइवेट लिमिटेड", ["lakshmee", "devalaparsa", "praaiveta", "limiteda"]),
            ("आनंद वेंचर्स", ["aananda", "vencharsa"]),
            ("ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍", ["shakti", "aagro", "limited"]),
            ("గ్రేట్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్", ["gret", "inpeks", "praivet", "limited"]),
            ("हॉस्पिटल एंड रिसर्च सेंटर", ["hospitala", "risarcha", "sentara"]),
            ("ऑटो टेक्नोलॉजीज", ["oto", "teknolojeeja"]),
            ("മാർക്കറ്റിംഗ്", ["maarkkarrrring"]),
        ]
        for indic_name, expected_tokens in cases:
            translit = transliterate_name(indic_name)
            self.assertTrue(len(translit) > 0)
            for tok in expected_tokens:
                self.assertIn(tok, translit, f"Expected '{tok}' in transliterated '{translit}' from '{indic_name}'")

    def test_house_number_and_postal_code(self):
        """Test extraction of house/building numbers and postal codes."""
        cases = [
            ("1795 Westchester Drive, High Point, NC 27262", "1795", "27262"),
            ("Plot No 126 Flat No 102 Rajiv Nagar, Hyderabad 500045", "126", "500045"),
            ("K-12, Shop-20, Second Floor Arcade, Jaipur, Rajasthan 302001", "K-12", "302001"),
            ("5 bis Rue Pierre Dignac, 33260 La Teste-de-Buch", "5", "33260"),
            ("NO. 5 ALLÉE DES HÊTRES, 44210 Pornic", "5", "44210"),
            ("18 RUE JEN ZAY, Dunkerque, 59140", "18", "59140"),
            ("6-2-101/5/C, Hyderabad, Telangana", "6-2-101/5/C", ""),
        ]
        for addr, exp_house, exp_postal in cases:
            h = extract_house_number(addr)
            p = extract_postal_code(addr)
            if exp_house:
                self.assertEqual(h, exp_house, f"House number mismatch for '{addr}': got '{h}', expected '{exp_house}'")
            if exp_postal:
                self.assertEqual(p, exp_postal, f"Postal code mismatch for '{addr}': got '{p}', expected '{exp_postal}'")

    def test_open_set_country(self):
        """Verify country is preserved as an open set without hardcoded restrictions."""
        for c in ["US", "India", "France", "Germany", "Brazil", "Japan", "UnknownCountry"]:
            rec = {
                "entity_id": "S1-999",
                "business_name": "Acme Global",
                "business_address": "123 Main St",
                "country": c
            }
            res = preprocess_record(rec)
            self.assertEqual(res["country"], c)

    def test_dataframe_preprocessing(self):
        """Test preprocessing on a pandas DataFrame with mixed countries and missing values."""
        data = {
            "entity_id": ["S1-001", "S2-002", "S3-003", "S1-004"],
            "business_name": [
                "Orelee's Barbershop Inc",
                "लक्ष्मी डेवलपर्स प्राइवेट लिमिटेड",
                "Thermal & Fils SASU",
                "Global Ventures"
            ],
            "business_address": [
                "1795 Westchester Dr, High Point, NC 27262",
                "H.No 1338, Kolhapur 416001",
                "20 Rue Parmentier, Dunkerque 59140",
                None
            ],
            "country": ["US", "India", "France", "Canada"]
        }
        df = pd.DataFrame(data)
        out_df = preprocess_dataframe(df)

        for col in ["entity_id", "business_name", "business_address", "country"]:
            self.assertIn(col, out_df.columns)

        expected_cols = [
            "name_normalized", "name_compact", "name_tokens", "name_without_legal_suffix",
            "name_transliterated", "address_normalized", "address_compact", "address_tokens",
            "postal_code", "numeric_tokens", "house_number"
        ]
        for col in expected_cols:
            self.assertIn(col, out_df.columns)

        self.assertEqual(out_df.loc[3, "address_normalized"], "")
        self.assertEqual(out_df.loc[3, "postal_code"], "")
        self.assertEqual(out_df.loc[3, "numeric_tokens"], ())
        self.assertIn("lakshmee", out_df.loc[1, "name_transliterated"])


if __name__ == "__main__":
    unittest.main()
