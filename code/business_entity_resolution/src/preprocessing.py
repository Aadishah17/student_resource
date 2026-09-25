"""
ML Challenge 2026: Business Entity Resolution
Module: preprocessing.py

Handles robust, memory-efficient data preprocessing and normalization
for noisy business entity records across multiple sources, scripts, and languages.

Features:
- Stream/chunk-based TSV reading and processing to protect memory (15.5 GB RAM constraint)
- Non-destructive processing: preserves all original columns intact
- Business name normalization, legal suffix stripping, and tokenization
- Safe, zero-dependency local Indic-script transliteration (ISCII Unicode relative offset mapping)
- Address normalization, abbreviation standardization, house number, postal code, and numeric token extraction
- Open-set country preservation
"""

import os
import re
import sys
import unicodedata
from typing import Dict, List, Tuple, Generator, Optional, Any
import pandas as pd

# ==============================================================================
# 1. Indic Script Local Transliteration (ISCII Offset Mapping)
# ==============================================================================

# Relative offsets in the Unicode standard for Indic scripts (ISCII alignment)
# Blocks: Devanagari (0x0900), Bengali (0x0980), Gurmukhi (0x0A00), Gujarati (0x0A80),
# Oriya (0x0B00), Tamil (0x0B80), Telugu (0x0C00), Kannada (0x0C80), Malayalam (0x0D00)
_INDIC_OFFSET_MAP: Dict[int, str] = {
    # Independent Vowels
    0x05: 'a', 0x06: 'aa', 0x07: 'i', 0x08: 'ee', 0x09: 'u', 0x0A: 'oo',
    0x0B: 'ri', 0x0E: 'e', 0x0F: 'e', 0x10: 'ai', 0x11: 'o', 0x12: 'o', 0x13: 'o', 0x14: 'au',
    # Consonants
    0x15: 'k', 0x16: 'kh', 0x17: 'g', 0x18: 'gh', 0x19: 'ng',
    0x1A: 'ch', 0x1B: 'chh', 0x1C: 'j', 0x1D: 'jh', 0x1E: 'ny',
    0x1F: 't', 0x20: 'th', 0x21: 'd', 0x22: 'dh', 0x23: 'n',
    0x24: 't', 0x25: 'th', 0x26: 'd', 0x27: 'dh', 0x28: 'n', 0x29: 'nn',
    0x2A: 'p', 0x2B: 'ph', 0x2C: 'b', 0x2D: 'bh', 0x2E: 'm',
    0x2F: 'y', 0x30: 'r', 0x31: 'rr', 0x32: 'l', 0x33: 'll', 0x34: 'lll',
    0x35: 'v', 0x36: 'sh', 0x37: 'sh', 0x38: 's', 0x39: 'h',
    # Nukta / additional consonants
    0x3C: '',  # Nukta (modifies preceding consonant, non-spacing)
    0x58: 'q', 0x59: 'kh', 0x5A: 'g', 0x5B: 'z', 0x5C: 'r', 0x5D: 'rh', 0x5E: 'f', 0x5F: 'y',
    # Malayalam Chillu pure consonants
    0x7A: 'nn', 0x7B: 'n', 0x7C: 'r', 0x7D: 'l', 0x7E: 'll',
    # Dependent Vowels (Matras)
    0x3E: 'aa', 0x3F: 'i', 0x40: 'ee', 0x41: 'u', 0x42: 'oo',
    0x43: 'ri', 0x46: 'e', 0x47: 'e', 0x48: 'ai', 0x49: 'o', 0x4A: 'o', 0x4B: 'o', 0x4C: 'au',
    # Diacritics & Signs
    0x01: 'n', # Candrabindu
    0x02: 'n', # Anusvara
    0x03: 'h', # Visarga
    0x70: 'n', # Gurmukhi Tippi
    0x4D: '',  # Virama/Halant (mutes the inherent vowel)
}


def has_indic_characters(text: str) -> bool:
    """Returns True if the text contains any Indic Unicode characters."""
    for ch in text:
        if 0x0900 <= ord(ch) <= 0x0D7F:
            return True
    return False


def transliterate_indic(text: str) -> str:
    """
    Transliterates text containing Indic scripts (Devanagari, Tamil, Telugu,
    Gujarati, Bengali, Odia, Kannada, Malayalam, Gurmukhi) into Latin phonetics.
    
    Safe, local, and zero-dependency implementation using ISCII Unicode alignment.
    If no Indic characters are present, returns the text unchanged.
    """
    if not text:
        return ""
    
    out: List[str] = []
    i = 0
    n = len(text)
    has_indic = False

    while i < n:
        ch = text[i]
        code = ord(ch)
        
        # Check if code falls within the South Asian Indic Unicode block range
        if 0x0900 <= code <= 0x0D7F:
            has_indic = True
            offset = code & 0x7F
            is_consonant = (0x15 <= offset <= 0x39) or (0x58 <= offset <= 0x5F)
            phoneme = _INDIC_OFFSET_MAP.get(offset, '')
            
            if is_consonant:
                # Check for attached matra or virama in the next character
                if i + 1 < n and 0x0900 <= ord(text[i + 1]) <= 0x0D7F:
                    next_offset = ord(text[i + 1]) & 0x7F
                    if next_offset == 0x4D:  # Virama/Halant: suppress inherent vowel
                        out.append(phoneme)
                        i += 2
                        continue
                    elif next_offset in _INDIC_OFFSET_MAP and (0x3E <= next_offset <= 0x4C):
                        # Dependent vowel sign (matra)
                        out.append(phoneme + _INDIC_OFFSET_MAP[next_offset])
                        i += 2
                        continue
                # Inherent vowel 'a'
                out.append(phoneme + 'a')
                i += 1
            else:
                out.append(phoneme)
                i += 1
        else:
            out.append(ch)
            i += 1

    return "".join(out) if has_indic else text


# ==============================================================================
# 2. Business Name Normalization Patterns
# ==============================================================================

# Noisy prefixes to strip from business names
_PREFIX_CLEAN_RE = re.compile(
    r'^(?:m/s\.?|m/s|ms\.?|smt\.?|shri\.?|mr\.?|mrs\.?|dr\.?|<<|>>|--|\*\*\*|#+)\s*',
    re.IGNORECASE
)

# Common legal suffixes across US, India, France, and international businesses
_LEGAL_SUFFIX_LIST = [
    # Multi-word suffixes
    'private limited', 'pvt ltd', 'pvt limited', 'private ltd', 'p limited',
    'limited liability company', 'limited liability partnership',
    'societe anonyme', 'societe a responsabilite limitee',
    'incorporated', 'corporation', 'proprietorship',
    'enterprise', 'enterprises', 'associates', 'technologies', 'solutions', 'consultants',
    'private', 'public', 'limited', 'company',
    # Single-word / acronyms with or without periods
    's.a.r.l.', 'sarl', 's.a.s.u.', 'sasu', 's.a.s.', 'sas', 'eurl', 'sci', 's.a.', 'sa',
    'l.l.c.', 'llc', 'l.l.p.', 'llp', 'ltd', 'inc', 'corp', 'p.c.', 'pc', 'co', 'plc', 'gmbh',
    # Indic script legal designations
    'प्राइवेट लिमिटेड', 'प्रा. लि.', 'लिमिटेड', 'प्रा लि',
    'ప్రైవేట్ లిమిటెడ్', 'లిమిటెడ్',
    'பிரைவேட் லிமிடெட்', 'லிமிடெட்',
    'પ્રાઇવેટ લિમિટેડ', 'લિમિટેડ',
    'প্রাইভেট লিমিটেড', 'লিমিটেড',
    'ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍', 'ଲିମିଟେଡ୍'
]

# Regex using boundary or space matching to work across both ASCII and Unicode scripts
_LEGAL_SUFFIX_RE = re.compile(
    r'(?:(?<=[\s\b])|^)(?:' + '|'.join(re.escape(s) for s in sorted(_LEGAL_SUFFIX_LIST, key=len, reverse=True)) + r')(?:(?=[\s\b])|$)',
    re.IGNORECASE
)


def strip_latin_accents(text: str) -> str:
    """
    Decomposes Latin accents (NFKD) while preserving non-Latin Unicode
    combining marks (such as Indic matras).
    """
    if not text:
        return ""
    if any(0x00C0 <= ord(c) <= 0x024F for c in text):
        nfkd = unicodedata.normalize('NFKD', text)
        # Only strip combining marks in the Latin diacritics range (0x0300 to 0x036F)
        return ''.join(c for c in nfkd if not (0x0300 <= ord(c) <= 0x036F))
    return text


def clean_unicode_text(text: str, keep_extra: str = "") -> str:
    """
    Retains all Unicode letters (L), numbers (N), and combining marks (M, e.g. Indic matras/viramas),
    replacing punctuation and symbols with spaces.
    """
    if not text:
        return ""
    chars = []
    for c in text:
        cat0 = unicodedata.category(c)[0]
        if cat0 in ('L', 'N', 'M') or c.isspace() or (keep_extra and c in keep_extra):
            chars.append(c)
        else:
            chars.append(' ')
    return ' '.join(''.join(chars).split())


def normalize_name(name: Any) -> str:
    """
    Standard business name normalization:
    - Lowercase
    - Unicode Latin accent normalization
    - Replace '&' with ' and '
    - Strip apostrophes to preserve contractions/possessives (e.g. Orelee's -> orelees)
    - Strip noisy prefixes (M/s, Smt, <<, --, ***)
    - Retain Unicode alphanumeric characters and Indic combining marks
    - Collapse repeated spaces
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""

    # Strip noisy leading symbols and honorific prefixes
    text = _PREFIX_CLEAN_RE.sub('', text)

    # Unicode Latin accent stripping
    text = strip_latin_accents(text)

    # Lowercase
    text = text.lower()

    # Replace '&' with ' and '
    text = text.replace('&', ' and ')

    # Strip apostrophes to keep contractions/possessives intact (e.g. Orelee's -> orelees)
    text = re.sub(r"['’`]", '', text)

    # Retain Unicode letters, numbers, and combining marks (matras)
    text = clean_unicode_text(text)
    return text


def remove_legal_suffix(name_norm: str) -> str:
    """
    Removes legal entity designations (Inc, LLC, Pvt Ltd, SARL, SAS, Corp, etc.)
    from normalized business name across Latin and Indic scripts.
    """
    if not name_norm:
        return ""
    cleaned = _LEGAL_SUFFIX_RE.sub(' ', name_norm)
    # Strip dangling trailing conjunctions (e.g. from "& Co" -> "and")
    cleaned = re.sub(r'\b(?:and|et)\b\s*$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else name_norm


def compact_string(text: str) -> str:
    """Returns alphanumeric characters and combining marks only, removing spaces and punctuation."""
    if not text:
        return ""
    return ''.join(c for c in text.lower() if unicodedata.category(c)[0] in ('L', 'N', 'M'))


def tokenize_string(text: str) -> Tuple[str, ...]:
    """Splits normalized text into a tuple of clean words."""
    if not text:
        return ()
    return tuple(text.split())


def transliterate_name(name: Any) -> str:
    """
    Creates a transliterated representation of the business name.
    If Indic characters exist, transliterates them to Latin phonetics first,
    then applies name normalization.
    If no Indic characters exist, returns standard normalized name.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    # Transliterate Indic characters to phonetic Latin
    translit = transliterate_indic(text)
    # Apply standard normalization
    return normalize_name(translit)


# ==============================================================================
# 3. Address Normalization Patterns
# ==============================================================================

# Mapping of common address abbreviations to standard forms
_ADDR_ABBREV_RULES = [
    (re.compile(r'\bst\b\.?', re.IGNORECASE), 'street'),
    (re.compile(r'\brd\b\.?', re.IGNORECASE), 'road'),
    (re.compile(r'\bave\b\.?|\bav\b\.?', re.IGNORECASE), 'avenue'),
    (re.compile(r'\bblvd\b\.?|\bbd\b\.?', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\bdr\b\.?', re.IGNORECASE), 'drive'),
    (re.compile(r'\bct\b\.?', re.IGNORECASE), 'court'),
    (re.compile(r'\bln\b\.?', re.IGNORECASE), 'lane'),
    (re.compile(r'\bpkwy\b\.?', re.IGNORECASE), 'parkway'),
    (re.compile(r'\bhwy\b\.?', re.IGNORECASE), 'highway'),
    (re.compile(r'\bfl\b\.?|\bflr\b\.?', re.IGNORECASE), 'floor'),
    (re.compile(r'\bste\b\.?', re.IGNORECASE), 'suite'),
    (re.compile(r'\bapt\b\.?', re.IGNORECASE), 'apartment'),
    (re.compile(r'\br\b\.?', re.IGNORECASE), 'rue'),
    (re.compile(r'\ball\b\.?', re.IGNORECASE), 'allee'),
    (re.compile(r'\bno\b\.?', re.IGNORECASE), 'number'),
    (re.compile(r'\bnr\b\.?', re.IGNORECASE), 'near'),
    (re.compile(r'\bopp\b\.?', re.IGNORECASE), 'opposite'),
    (re.compile(r'\bsec\b\.?', re.IGNORECASE), 'sector'),
    (re.compile(r'\bind\b\.?', re.IGNORECASE), 'industrial'),
    (re.compile(r'\best\b\.?', re.IGNORECASE), 'estate'),
    (re.compile(r'\bdist\b\.?', re.IGNORECASE), 'district'),
]

_NOISY_ADDR_TOKENS_RE = re.compile(r'\b(?:null|undefined|none)\b|[#*]+', re.IGNORECASE)


def normalize_address(address: Any) -> str:
    """
    Standard address normalization:
    - Safe handling of missing/NaN values
    - Lowercase and Unicode accent stripping
    - Removal of 'null', 'undefined', '#', and punctuation noise
    - Standard abbreviation expansion (St -> street, Rd -> road, etc.)
    """
    if address is None or (isinstance(address, float) and pd.isna(address)):
        return ""
    text = str(address).strip()
    if not text:
        return ""

    # Remove accents
    text = strip_latin_accents(text)

    # Lowercase
    text = text.lower()

    # Remove noise tokens (null, #, ***)
    text = _NOISY_ADDR_TOKENS_RE.sub(' ', text)

    # Standardize abbreviations
    for pattern, replacement in _ADDR_ABBREV_RULES:
        text = pattern.sub(replacement, text)

    # Retain Unicode letters, numbers, and hyphens/slashes in building numbers
    text = clean_unicode_text(text, keep_extra="-/")
    return text


def extract_postal_code(address: Any) -> str:
    """
    Extracts a postal code from an address string if present.
    Supports 6-digit PIN codes (India) and 5-digit ZIP / Code Postal (US, France).
    Returns an empty string if no valid postal code is found.
    """
    if address is None or (isinstance(address, float) and pd.isna(address)):
        return ""
    text = str(address)
    
    # Priority 1: 6-digit Indian PIN code (100000 to 999999)
    pin_matches = re.findall(r'\b([1-9][0-9]{5})\b', text)
    if pin_matches:
        return pin_matches[-1]

    # Priority 2: 5-digit US ZIP code or French Code Postal
    zip_matches = re.findall(r'\b([0-9]{5})\b', text)
    if zip_matches:
        return zip_matches[-1]

    return ""


def extract_numeric_tokens(address: Any) -> Tuple[str, ...]:
    """
    Extracts all numeric sequences from an address as an ordered tuple of strings.
    Useful as strong invariant anchors for candidate blocking and verification.
    """
    if address is None or (isinstance(address, float) and pd.isna(address)):
        return ()
    nums = re.findall(r'\b[0-9]+\b', str(address))
    return tuple(nums)


def extract_house_number(address: Any) -> str:
    """
    Extracts the primary building / house / plot / flat number from an address.
    Handles explicit prefixes (Plot No, H.No, Shop No, Unit, etc.) as well as
    leading number patterns.
    """
    if address is None or (isinstance(address, float) and pd.isna(address)):
        return ""
    text = str(address).strip()
    if not text:
        return ""

    # Check for explicit prefixed building indicators
    m = re.search(
        r'\b(?:plot\s*no\.?|h\.?\s*no\.?|house\s*no\.?|flat\s*no\.?|shop\s*no\.?|unit\s*no\.?|s\s*no\.?|no\.?|#)\s*([a-zA-Z0-9\-/]+)',
        text,
        re.IGNORECASE
    )
    if m:
        val = m.group(1).strip()
        val = re.sub(r'^[#\-/\s]+|[#\-/\s]+$', '', val)
        val = re.sub(r'#+', '', val)
        if any(c.isdigit() for c in val):
            return val

    # Check if address starts with a building / house number (e.g. "1795 Westchester Dr")
    m2 = re.match(r'^\s*([a-zA-Z]?[0-9]+[a-zA-Z0-9\-/]*)', text)
    if m2:
        val = m2.group(1).strip()
        val = re.sub(r'^[#\-/\s]+|[#\-/\s]+$', '', val)
        if any(c.isdigit() for c in val):
            return val

    # Fallback: inspect early tokens for digit-bearing identifiers
    tokens = text.replace(',', ' ').split()
    for tok in tokens[:4]:
        clean_tok = tok.strip("#()[].,/-")
        if any(c.isdigit() for c in clean_tok) and len(clean_tok) <= 10:
            # Skip if it is likely a 5- or 6-digit postal code
            if not (len(clean_tok) in (5, 6) and clean_tok.isdigit()):
                return clean_tok

    return ""


# ==============================================================================
# 4. Record-Level Preprocessor
# ==============================================================================

def preprocess_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single business record dictionary.
    Preserves all original fields and adds all required normalized representations.
    
    Country is treated as an open-set string.
    """
    entity_id = record.get("entity_id", "")
    raw_name = record.get("business_name", "")
    raw_addr = record.get("business_address", "")
    country = str(record.get("country", "")).strip()

    # Name representations
    name_norm = normalize_name(raw_name)
    name_comp = compact_string(name_norm)
    name_toks = tokenize_string(name_norm)
    name_no_legal = remove_legal_suffix(name_norm)
    name_trans = transliterate_name(raw_name)

    # Address representations
    addr_norm = normalize_address(raw_addr)
    addr_comp = compact_string(addr_norm)
    addr_toks = tokenize_string(addr_norm)
    postal_code = extract_postal_code(raw_addr)
    num_toks = extract_numeric_tokens(raw_addr)
    house_num = extract_house_number(raw_addr)

    return {
        # Original columns preserved exactly
        "entity_id": entity_id,
        "business_name": raw_name,
        "business_address": raw_addr,
        "country": country,
        # Derived business name representations
        "name_normalized": name_norm,
        "name_compact": name_comp,
        "name_tokens": name_toks,
        "name_without_legal_suffix": name_no_legal,
        "name_transliterated": name_trans,
        # Derived address representations
        "address_normalized": addr_norm,
        "address_compact": addr_comp,
        "address_tokens": addr_toks,
        "postal_code": postal_code,
        "numeric_tokens": num_toks,
        "house_number": house_num,
    }


# ==============================================================================
# 5. DataFrame and Chunked / Streaming Processing (Memory Safe)
# ==============================================================================

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies batch preprocessing to a pandas DataFrame.
    Preserves original columns and appends all derived feature columns.
    """
    out_df = df.copy()

    # Fill NaNs safely
    out_df["business_name"] = out_df["business_name"].fillna("").astype(str)
    out_df["business_address"] = out_df["business_address"].fillna("").astype(str)
    out_df["country"] = out_df["country"].fillna("").astype(str)

    # Name transformations
    out_df["name_normalized"] = [normalize_name(x) for x in out_df["business_name"]]
    out_df["name_compact"] = [compact_string(x) for x in out_df["name_normalized"]]
    out_df["name_tokens"] = [tokenize_string(x) for x in out_df["name_normalized"]]
    out_df["name_without_legal_suffix"] = [remove_legal_suffix(x) for x in out_df["name_normalized"]]
    out_df["name_transliterated"] = [transliterate_name(x) for x in out_df["business_name"]]

    # Address transformations
    out_df["address_normalized"] = [normalize_address(x) for x in out_df["business_address"]]
    out_df["address_compact"] = [compact_string(x) for x in out_df["address_normalized"]]
    out_df["address_tokens"] = [tokenize_string(x) for x in out_df["address_normalized"]]
    out_df["postal_code"] = [extract_postal_code(x) for x in out_df["business_address"]]
    out_df["numeric_tokens"] = [extract_numeric_tokens(x) for x in out_df["business_address"]]
    out_df["house_number"] = [extract_house_number(x) for x in out_df["business_address"]]

    return out_df


def stream_tsv_preprocessed(
    tsv_path: str,
    chunksize: int = 100_000
) -> Generator[pd.DataFrame, None, None]:
    """
    Streams a large TSV file in chunks, yielding preprocessed DataFrames.
    Guarantees constant memory consumption regardless of file size.
    """
    for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=chunksize, dtype=str):
        yield preprocess_dataframe(chunk)
