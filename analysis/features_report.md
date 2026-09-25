# Pairwise Feature Engineering Report: Business Entity Resolution

**Author**: Senior ML Engineer & Entity Resolution Specialist  
**Module**: [`code/business_entity_resolution/src/features.py`](file:///C:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/code/business_entity_resolution/src/features.py)  
**Test Suite**: [`code/business_entity_resolution/tests/test_features.py`](file:///C:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/code/business_entity_resolution/tests/test_features.py) | [`tests/test_features.py`](file:///C:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/tests/test_features.py)  
**Benchmark Runner**: [`code/business_entity_resolution/src/run_features_benchmark.py`](file:///C:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/code/business_entity_resolution/src/run_features_benchmark.py)  
**Evaluation Scope**: 57,108 candidate pairs generated under Configuration H across a stratified validation cohort of 1,000 Source 1 entities against a pool of 103,516 target records.  

---

## 1. Executive Summary

Following candidate generation under **Configuration H** (which achieves **$97.35\%$ overall recall** and **$87.15\%$ cross-script recall** with an average of $55.69$ candidates per S1 entity), pairwise feature engineering computes the discriminative signals required for subsequent binary classification and ranking.

The feature engineering module [`features.py`](file:///C:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/code/business_entity_resolution/src/features.py) extracts **96 dense numeric features** across six distinct functional groups for every $(S1 \to \text{candidate } S2/S3)$ pair.

### Benchmark Highlights on Validation Candidate Pairs ($N = 57,108$):
- **Feature Vector Dimensionality**: Exactly **96 numeric features**
- **Data Type**: `float32` (memory efficient and directly consumable by LightGBM, XGBoost, and CatBoost)
- **Data Integrity**: **0 NaN values**, **0 infinity values**, 100% finite real numbers
- **Memory Consumption**: Dense feature matrix of 57,108 pairs occupies only **$20.91$ MB**
- **Peak Process Memory**: **$3,081$ MB ($3.01$ GB)**, well within the 15.5 GB RAM constraint
- **Feature Extraction Throughput**: **$4,904.2$ pairs / second** ($11.64$ seconds total for 57,108 pairs)
- **External Lookups**: **Strictly 0 external API calls, 0 internet lookups, 0 geocoding queries**

---

## 2. Feature Architecture & Group Taxonomy

The 96 features are organized into six cohesive groups, addressing both lexical and semantic variations across domestic and multilingual cross-script records:

```mermaid
graph TD
    Pair[Candidate Pair: S1 & Candidate S2/S3] --> G1[Group 1: Name Features - 45 Feats]
    Pair --> G2[Group 2: Address Features - 22 Feats]
    Pair --> G3[Group 3: Cross-Field Interactions - 10 Feats]
    Pair --> G4[Group 4: Script & Language - 6 Feats]
    Pair --> G5[Group 5: Source Attribution - 2 Feats]
    Pair --> G6[Group 6: Blocking Evidence - 11 Feats]
    
    G1 --> Matrix[Dense Feature Matrix: N x 96 float32]
    G2 --> Matrix
    G3 --> Matrix
    G4 --> Matrix
    G5 --> Matrix
    G6 --> Matrix
```

### Group Breakdown:

### Group 1: Business Name Features (45 features)
Evaluates normalized names, compact names, legal suffix-stripped names, transliterated names, and consonant skeleton representations:
1. **Exact Matches**:
   - `name_exact`: Exact equality of normalized names
   - `name_compact_exact`: Exact equality of compact strings (no spaces or punctuation)
   - `name_nosuff_exact`: Exact equality of names with legal suffixes removed
   - `name_translit_exact`: Exact match between Latin S1 name and transliterated Indic candidate name
   - `name_nosuff_translit_exact`: Exact match after stripping legal suffixes from transliterated string
2. **Fuzzy String Similarities**:
   - `name_levenshtein_ratio`, `name_normalized_levenshtein`
   - `name_token_sort_ratio`: Word-order invariant similarity
   - `name_token_set_ratio`: Subset/superset token similarity
   - `name_w_ratio`: Weighted fuzzy ratio
   - Transliterated counterparts: `name_translit_levenshtein_ratio`, `name_translit_normalized_levenshtein`, `name_translit_token_sort_ratio`, `name_translit_token_set_ratio`, `name_translit_w_ratio`
3. **Character N-Gram Overlaps**:
   - Character 2, 3, 4, 5-gram Jaccard on normalized names (`name_char2_jaccard` through `name_char5_jaccard`)
   - Character 2, 3, 4-gram Jaccard on transliterated names (`name_translit_char2_jaccard` through `name_translit_char4_jaccard`)
4. **Token Features**:
   - Word token Jaccard (`name_token_jaccard`, `name_translit_token_jaccard`)
   - Word token overlap counts (`name_token_overlap_count`, `name_translit_token_overlap_count`)
   - Shared token fraction relative to $\min$ and $\max$ token lengths
5. **Structural Features**:
   - Character length difference and ratio (`name_length_diff`, `name_length_ratio`)
   - Word token count difference and ratio (`name_token_count_diff`, `name_token_count_ratio`)
   - First-token and last-token exact equality (`name_first_token_exact`, `name_last_token_exact`)
   - Prefix 3, 4, 5 character matches (`name_prefix3_exact` through `name_prefix5_exact`)
   - Transliterated first token and prefix 3 matches
6. **Consonant Skeletons (Vowel-Invariant Phonetic Bridge)**:
   - `consonant_skeleton_exact`: Exact match of consonant skeletons
   - `consonant_skeleton_levenshtein`: Normalized edit similarity on consonant skeletons
   - `consonant_skeleton_jaccard`: Character 2-gram overlap on consonant skeletons
   - `consonant_skeleton_len_diff`, `consonant_skeleton_len_ratio`

---

### Group 2: Address Features & Explicit Missingness Handling (22 features)
1. **Address Similarities**:
   - `address_exact`, `address_compact_similarity`, `address_levenshtein_ratio`, `address_normalized_levenshtein`
   - `address_token_jaccard`, `address_token_set_ratio`, `address_char3_ngram_jaccard`
   - `address_length_ratio`, `address_token_count_ratio`
2. **Numeric Token Overlap**:
   - `address_numeric_token_overlap`: Count of shared numeric tokens (house, suite, plot, sector, PIN)
   - `address_numeric_token_jaccard`: Jaccard similarity across numeric tokens
3. **Structured Components**:
   - `house_number_exact`: Exact match on building/house/plot numbers
   - `postal_code_exact`: Exact match on 5-digit US ZIP or 6-digit Indian PIN
   - `postal_code_prefix3_exact`: Match on first 3 digits (sectional center or postal district)
4. **Explicit Missingness Indicators**:
   - When an address is missing in raw data (13.07% of target records), computing similarity directly would generate misleading zero scores indistinguishable from true non-matches, or worse, impute artificial similarities.
   - We explicitly construct boolean indicators:
     * `s1_address_missing`, `candidate_address_missing`, `both_address_missing`, `either_address_missing`
     * `postal_missing_either`, `postal_missing_both`
     * `house_missing_either`, `house_missing_both`
   - When either address is missing, address similarity features are cleanly clamped to $0.0$, allowing decision tree split points to isolate missingness explicitly without distorting similarity gradients.

---

### Group 3: Cross-Field Interaction Features (10 features)
Captures non-linear combinations where address anchors validate name similarities and vice versa:
- `same_country`: Binary indicator ($1.0$ if countries match)
- `same_postal_and_high_name_sim`: $1.0$ if postal codes match AND normalized name Levenshtein $\ge 0.70$
- `same_house_and_high_name_sim`: $1.0$ if house numbers match AND normalized name Levenshtein $\ge 0.70$
- `same_postal_and_same_house`: $1.0$ if both postal code and house number match exactly
- `same_postal_and_same_name_prefix`: $1.0$ if postal code matches AND first 3 letters of name match
- `addr_sim_x_name_sim`: Multiplicative product of `address_token_jaccard` and `name_normalized_levenshtein`
- `max_name_similarity`: Maximum similarity score across all name metrics
- `max_address_similarity`: Maximum similarity score across all address metrics
- `name_and_addr_high_confidence`: $1.0$ if `max_name_similarity` $\ge 0.80$ AND `max_address_similarity` $\ge 0.60$
- `name_exact_diff_address`: Homonym / franchise flag ($1.0$ if name matches exactly, but addresses are non-empty and have token Jaccard $< 0.20$)

---

### Group 4: Script & Language Signals (6 features)
- `s1_has_indic`, `candidate_is_indic`: Indicators of non-Latin script in name or address
- `is_cross_script`: $1.0$ if one record is Latin and the other is Indic
- `name_has_transliteration`: $1.0$ if transliterated representation differs from raw string
- `s1_script_latin`, `candidate_script_latin`: Script type indicators

---

### Group 5: Source Attribution Signals (2 features)
- `candidate_source_is_S2`: $1.0$ if candidate originates from Source 2
- `candidate_source_is_S3`: $1.0$ if candidate originates from Source 3  
Enables the tree-based model to learn source-specific noise properties (e.g. OCR artifacts in S2 vs web scrape artifacts in S3).

---

### Group 6: Multi-Rule Blocking Evidence (11 features)
Grounds the classifier in the candidate generation provenance:
- `blocked_exact_name`, `blocked_compact_name`, `blocked_suffix_name`, `blocked_translit`
- `blocked_postal`, `blocked_house`, `blocked_address_token`
- `blocked_char_ngram`, `blocked_consonant_skeleton`, `blocked_address_name_combo`
- `num_blocking_rules`: Integer count ($1$ to $10$) of independent blocking rules that generated this candidate pair. Candidate pairs recovered independently by 4+ rules have an empirical true-match precision exceeding $85\%$.

---

## 3. Complete Feature Registry & Distribution Statistics

Evaluated across $57,108$ candidate pairs extracted from the representative validation sample:

| Idx | Feature Name | Mean | Min | Max | NonZero% | Description |
|---|---|---|---|---|---|---|
| 0 | `name_exact` | 0.0179 | 0.00 | 1.00 | 1.79% | Exact normalized name equality |
| 1 | `name_compact_exact` | 0.0187 | 0.00 | 1.00 | 1.87% | Exact compact name equality |
| 2 | `name_nosuff_exact` | 0.0374 | 0.00 | 1.00 | 3.74% | Exact suffix-stripped name equality |
| 3 | `name_translit_exact` | 0.0180 | 0.00 | 1.00 | 1.80% | Exact transliterated name equality |
| 4 | `name_nosuff_translit_exact` | 0.0376 | 0.00 | 1.00 | 3.76% | Exact transliterated suffix-stripped name |
| 5 | `name_levenshtein_ratio` | 0.4552 | 0.00 | 1.00 | 99.96% | Levenshtein similarity ratio on names |
| 6 | `name_normalized_levenshtein` | 0.3315 | 0.00 | 1.00 | 99.59% | Normalized edit distance similarity |
| 7 | `name_token_sort_ratio` | 0.4437 | 0.00 | 1.00 | 99.96% | Token sort ratio (word order invariant) |
| 8 | `name_token_set_ratio` | 0.4916 | 0.00 | 1.00 | 99.96% | Token set ratio (subset invariant) |
| 9 | `name_w_ratio` | 0.5370 | 0.00 | 1.00 | 99.96% | Weighted fuzzy ratio |
| 10 | `name_translit_levenshtein_ratio` | 0.4709 | 0.00 | 1.00 | 99.96% | Levenshtein ratio on transliterated name |
| 11 | `name_translit_normalized_levenshtein`| 0.3433 | 0.00 | 1.00 | 99.74% | Normalized edit distance on transliterated |
| 12 | `name_translit_token_sort_ratio` | 0.4583 | 0.00 | 1.00 | 99.96% | Token sort ratio on transliterated name |
| 13 | `name_translit_token_set_ratio` | 0.5062 | 0.00 | 1.00 | 99.96% | Token set ratio on transliterated name |
| 14 | `name_translit_w_ratio` | 0.5535 | 0.00 | 1.00 | 99.96% | WRatio on transliterated name |
| 15 | `name_char2_jaccard` | 0.2176 | 0.00 | 1.00 | 88.13% | Character 2-gram Jaccard overlap |
| 16 | `name_char3_jaccard` | 0.1601 | 0.00 | 1.00 | 60.64% | Character 3-gram Jaccard overlap |
| 17 | `name_char4_jaccard` | 0.1397 | 0.00 | 1.00 | 51.16% | Character 4-gram Jaccard overlap |
| 18 | `name_char5_jaccard` | 0.1238 | 0.00 | 1.00 | 46.52% | Character 5-gram Jaccard overlap |
| 19 | `name_translit_char2_jaccard` | 0.2253 | 0.00 | 1.00 | 91.91% | Character 2-gram Jaccard on transliterated |
| 20 | `name_translit_char3_jaccard` | 0.1638 | 0.00 | 1.00 | 63.47% | Character 3-gram Jaccard on transliterated |
| 21 | `name_translit_char4_jaccard` | 0.1422 | 0.00 | 1.00 | 53.51% | Character 4-gram Jaccard on transliterated |
| 22 | `name_token_jaccard` | 0.1461 | 0.00 | 1.00 | 46.01% | Word token Jaccard similarity |
| 23 | `name_token_overlap_count` | 0.7321 | 0.00 | 8.00 | 46.01% | Number of shared word tokens |
| 24 | `name_shared_token_fraction_min` | 0.2389 | 0.00 | 1.00 | 46.01% | Shared token fraction / min tokens |
| 25 | `name_shared_token_fraction_max` | 0.1866 | 0.00 | 1.00 | 46.01% | Shared token fraction / max tokens |
| 26 | `name_translit_token_jaccard` | 0.1469 | 0.00 | 1.00 | 46.49% | Transliterated token Jaccard |
| 27 | `name_translit_token_overlap_count` | 0.7373 | 0.00 | 8.00 | 46.49% | Transliterated shared token count |
| 28 | `name_translit_shared_token_fraction_min`| 0.2404 | 0.00 | 1.00 | 46.49% | Transliterated token fraction / min |
| 29 | `name_length_diff` | 8.5117 | 0.00 | 56.00 | 93.82% | Character length absolute difference |
| 30 | `name_length_ratio` | 0.7153 | 0.04 | 1.00 | 100.00% | Character length ratio (min / max) |
| 31 | `name_token_count_diff` | 1.1406 | 0.00 | 11.00 | 69.94% | Word token count difference |
| 32 | `name_token_count_ratio` | 0.7423 | 0.10 | 1.00 | 100.00% | Word token count ratio (min / max) |
| 33 | `name_first_token_exact` | 0.2681 | 0.00 | 1.00 | 26.81% | First word token exact equality |
| 34 | `name_last_token_exact` | 0.1308 | 0.00 | 1.00 | 13.08% | Last word token exact equality |
| 35 | `name_prefix3_exact` | 0.2962 | 0.00 | 1.00 | 29.62% | First 3 characters exact match |
| 36 | `name_prefix4_exact` | 0.2934 | 0.00 | 1.00 | 29.34% | First 4 characters exact match |
| 37 | `name_prefix5_exact` | 0.2905 | 0.00 | 1.00 | 29.05% | First 5 characters exact match |
| 38 | `name_translit_first_token_exact` | 0.2702 | 0.00 | 1.00 | 27.02% | First transliterated token exact match |
| 39 | `name_translit_prefix3_exact` | 0.3030 | 0.00 | 1.00 | 30.30% | Transliterated prefix 3 chars match |
| 40 | `consonant_skeleton_exact` | 0.0407 | 0.00 | 1.00 | 4.07% | Consonant skeleton exact equality |
| 41 | `consonant_skeleton_levenshtein` | 0.3305 | 0.00 | 1.00 | 94.24% | Consonant skeleton normalized edit distance |
| 42 | `consonant_skeleton_jaccard` | 0.2004 | 0.00 | 1.00 | 66.78% | Consonant skeleton 2-gram overlap |
| 43 | `consonant_skeleton_len_diff` | 4.2047 | 0.00 | 28.00 | 88.23% | Consonant skeleton length difference |
| 44 | `consonant_skeleton_len_ratio` | 0.7002 | 0.00 | 1.00 | 99.92% | Consonant skeleton length ratio |
| 45 | `address_exact` | 0.0066 | 0.00 | 1.00 | 0.66% | Exact normalized address match |
| 46 | `address_compact_similarity` | 0.2605 | 0.00 | 1.00 | 98.26% | Compact address edit similarity |
| 47 | `address_levenshtein_ratio` | 0.4309 | 0.00 | 1.00 | 98.34% | Address Levenshtein ratio |
| 48 | `address_normalized_levenshtein` | 0.2892 | 0.00 | 1.00 | 98.34% | Normalized edit distance on address |
| 49 | `address_token_jaccard` | 0.1225 | 0.00 | 1.00 | 70.60% | Address word token Jaccard |
| 50 | `address_token_set_ratio` | 0.4778 | 0.00 | 1.00 | 98.34% | Address token set similarity |
| 51 | `address_char3_ngram_jaccard` | 0.1342 | 0.00 | 1.00 | 84.64% | Address character 3-gram Jaccard |
| 52 | `address_length_ratio` | 0.7728 | 0.00 | 1.00 | 98.34% | Address string length ratio |
| 53 | `address_token_count_ratio` | 0.7920 | 0.00 | 1.00 | 98.34% | Address token count ratio |
| 54 | `address_numeric_token_overlap` | 0.1975 | 0.00 | 8.00 | 18.02% | Matching numeric tokens in address |
| 55 | `address_numeric_token_jaccard` | 0.1476 | 0.00 | 1.00 | 19.06% | Jaccard overlap on numeric tokens |
| 56 | `house_number_exact` | 0.1620 | 0.00 | 1.00 | 16.20% | House/plot number exact equality |
| 57 | `postal_code_exact` | 0.0033 | 0.00 | 1.00 | 0.33% | Postal code exact equality |
| 58 | `postal_code_prefix3_exact` | 0.0037 | 0.00 | 1.00 | 0.37% | Postal code first 3 digits match |
| 59 | `s1_address_missing` | 0.0000 | 0.00 | 0.00 | 0.00% | Source 1 address is missing/empty |
| 60 | `candidate_address_missing` | 0.0166 | 0.00 | 1.00 | 1.66% | Target candidate address is missing |
| 61 | `both_address_missing` | 0.0000 | 0.00 | 0.00 | 0.00% | Both addresses are missing |
| 62 | `either_address_missing` | 0.0166 | 0.00 | 1.00 | 1.66% | Either address is missing |
| 63 | `postal_missing_either` | 0.9859 | 0.00 | 1.00 | 98.59% | Either postal code is missing |
| 64 | `postal_missing_both` | 0.8797 | 0.00 | 1.00 | 87.97% | Both postal codes are missing |
| 65 | `house_missing_either` | 0.1479 | 0.00 | 1.00 | 14.79% | Either house number is missing |
| 66 | `house_missing_both` | 0.0112 | 0.00 | 1.00 | 1.12% | Both house numbers are missing |
| 67 | `same_country` | 1.0000 | 1.00 | 1.00 | 100.00% | Both records share the same country |
| 68 | `same_postal_and_high_name_sim` | 0.0020 | 0.00 | 1.00 | 0.20% | Postal match + name similarity >= 0.7 |
| 69 | `same_house_and_high_name_sim` | 0.0221 | 0.00 | 1.00 | 2.21% | House match + name similarity >= 0.7 |
| 70 | `same_postal_and_same_house` | 0.0024 | 0.00 | 1.00 | 0.24% | Both postal code and house match |
| 71 | `same_postal_and_same_name_prefix` | 0.0025 | 0.00 | 1.00 | 0.25% | Postal match + name 3-char prefix match |
| 72 | `addr_sim_x_name_sim` | 0.0467 | 0.00 | 1.00 | 70.21% | Product of address Jaccard and name Lev |
| 73 | `max_name_similarity` | 0.5606 | 0.00 | 1.00 | 99.96% | Maximum similarity across all name metrics |
| 74 | `max_address_similarity` | 0.4778 | 0.00 | 1.00 | 98.34% | Maximum similarity across address metrics |
| 75 | `name_and_addr_high_confidence` | 0.0576 | 0.00 | 1.00 | 5.76% | Max name >= 0.8 and max address >= 0.6 |
| 76 | `name_exact_diff_address` | 0.0020 | 0.00 | 1.00 | 0.20% | Homonym flag (name match, address !=) |
| 77 | `s1_has_indic` | 0.0000 | 0.00 | 0.00 | 0.00% | S1 has Indic script characters |
| 78 | `candidate_is_indic` | 0.1054 | 0.00 | 1.00 | 10.54% | Candidate has Indic script characters |
| 79 | `is_cross_script` | 0.1054 | 0.00 | 1.00 | 10.54% | Pair bridges Latin and Indic scripts |
| 80 | `name_has_transliteration` | 0.0423 | 0.00 | 1.00 | 4.23% | Transliterated name != raw name |
| 81 | `s1_script_latin` | 1.0000 | 1.00 | 1.00 | 100.00% | S1 has Latin script |
| 82 | `candidate_script_latin` | 0.8946 | 0.00 | 1.00 | 89.46% | Candidate has Latin script |
| 83 | `candidate_source_is_S2` | 0.4985 | 0.00 | 1.00 | 49.85% | Candidate originates from Source 2 |
| 84 | `candidate_source_is_S3` | 0.5015 | 0.00 | 1.00 | 50.15% | Candidate originates from Source 3 |
| 85 | `blocked_exact_name` | 0.0179 | 0.00 | 1.00 | 1.79% | Blocking rule: exact normalized name |
| 86 | `blocked_compact_name` | 0.0187 | 0.00 | 1.00 | 1.87% | Blocking rule: exact compact name |
| 87 | `blocked_suffix_name` | 0.0374 | 0.00 | 1.00 | 3.74% | Blocking rule: suffix-stripped name |
| 88 | `blocked_translit` | 0.0035 | 0.00 | 1.00 | 0.35% | Blocking rule: transliterated name |
| 89 | `blocked_postal` | 0.0025 | 0.00 | 1.00 | 0.25% | Blocking rule: postal compound keys |
| 90 | `blocked_house` | 0.1549 | 0.00 | 1.00 | 15.49% | Blocking rule: house number compound keys |
| 91 | `blocked_address_token` | 0.5086 | 0.00 | 1.00 | 50.86% | Blocking rule: rare address tokens |
| 92 | `blocked_char_ngram` | 0.1739 | 0.00 | 1.00 | 17.39% | Blocking rule: character n-gram TF-IDF |
| 93 | `blocked_consonant_skeleton` | 0.2976 | 0.00 | 1.00 | 29.76% | Blocking rule: consonant skeleton |
| 94 | `blocked_address_name_combo` | 0.0506 | 0.00 | 1.00 | 5.06% | Blocking rule: address token + name prefix |
| 95 | `num_blocking_rules` | 1.2657 | 1.00 | 9.00 | 100.00% | Total blocking rules generating pair |

---

## 4. Concrete Example Feature Vectors

The following 8 case studies from the validation benchmark demonstrate how the feature vectors differentiate true positive matches from negative distractors across varied entity archetypes:

### Case 1: Exact Name & Address Concordance (True Positive)
- **S1**: `[S1-925783039] 'Orelee's Barbershop' | Addr: '1795 Westchester Drive, High Point, NC...' | US`
- **Target**: `[S2-157377754] 'Orelee's Barbershop' | Addr: '1795 WESTCHESTER DRIVE, NC, HIGH POINT...' | US`
- **Classification**: **TRUE MATCH (Ground Truth Positive)**
- **Generating Blocking Rules**: 8 distinct rules (`exact_name`, `compact_name`, `suffix_name`, `house`, `addr_token`, `char_ngram`, `consonant_skel`, `addr_name_combo`)
- **Key Extracted Features**:
  * `name_exact`: **`1.0000`** | `name_nosuff_exact`: **`1.0000`** | `name_normalized_levenshtein`: **`1.0000`**
  * `address_exact`: `0.0000` (formatting difference: lowercase vs uppercase) | `address_normalized_levenshtein`: **`0.8333`**
  * `address_token_jaccard`: **`1.0000`** | `house_number_exact`: **`1.0000`** (`1795`)
  * `num_blocking_rules`: **`8.0000`**
  * *Profile*: Strong signal across name, address, house number, and blocking provenance.

### Case 2: Legal Suffix Variation (True Positive)
- **S1**: `[S1-773889195] 'Prime Money' | Addr: '17560 Ellis Road, Tahlequah, OK...' | US`
- **Target**: `[S3-622232873] 'Prime Money (Inc)' | Addr: '17560 Ellis Rd, Cherokee County, Oklahoma...' | US`
- **Classification**: **TRUE MATCH (Ground Truth Positive)**
- **Generating Blocking Rules**: 7 distinct rules (`suffix_name`, `postal`, `house`, `addr_token`, `char_ngram`, `consonant_skel`, `addr_name_combo`)
- **Key Extracted Features**:
  * `name_exact`: `0.0000` | `name_nosuff_exact`: **`1.0000`** (suffix `(Inc)` stripped)
  * `name_token_sort_ratio`: **`0.8462`** | `consonant_skeleton_exact`: **`1.0000`** (`prmmn`)
  * `house_number_exact`: **`1.0000`** (`17560`) | `postal_code_exact`: **`1.0000`**
  * `num_blocking_rules`: **`7.0000`**
  * *Profile*: The `name_nosuff_exact` and `consonant_skeleton_exact` features bridge the legal suffix gap, while address numbers provide matching evidence.

### Case 3: Typo / Punctuation Noise (True Positive)
- **S1**: `[S1-773889195] 'Prime Money' | Addr: '17560 Ellis Road, Tahlequah, OK...' | US`
- **Target**: `[S2-970528089] '@primemoney' | Addr: 'TAHLEQUAH, OK, 0017560 ELLIS ROAD...' | US`
- **Classification**: **TRUE MATCH (Ground Truth Positive)**
- **Generating Blocking Rules**: 5 distinct rules (`compact_name`, `addr_token`, `char_ngram`, `consonant_skel`, `addr_name_combo`)
- **Key Extracted Features**:
  * `name_exact`: `0.0000` | `name_normalized_levenshtein`: **`0.9091`**
  * `consonant_skeleton_exact`: **`1.0000`** (`prmmn`)
  * `address_token_jaccard`: **`0.6667`** (`ellis road tahlequah ok`)
  * `num_blocking_rules`: **`5.0000`**
  * *Profile*: Consonant skeleton and compact normalization resolve the leading symbol (`@`) and space removal.

### Case 4: Latin $\to$ Indic Cross-Script Alignment (True Positive)
- **S1**: `[S1-755362802] 'Prabhav Business Center' | Addr: '797, Lake Town Block A, Kolkata, West Bengal...' | India`
- **Target**: `[S3-440254853] 'Mr Prabhav Business Services' | Addr: 'Plot 334 797, Lake Town Block A, Kolkata...' | India`
- **Transliterated Target**: `'prabhav business services'`
- **Classification**: **TRUE MATCH (Ground Truth Positive)**
- **Generating Blocking Rules**: 2 rules (`char_ngram`, `consonant_skel`)
- **Key Extracted Features**:
  * `is_cross_script`: **`1.0000`** | `candidate_is_indic`: **`1.0000`**
  * `name_exact`: `0.0000` | `name_translit_exact`: `0.0000`
  * `name_normalized_levenshtein`: **`0.7600`** | `name_char3_jaccard`: **`0.5172`**
  * `address_normalized_levenshtein`: **`0.6429`** | `address_token_jaccard`: **`0.5833`**
  * `num_blocking_rules`: **`2.0000`**
  * *Profile*: Cross-script indicator activates, transliterated similarities score high ($>0.75$), and address tokens (`lake town block a kolkata 797`) reinforce the match.

### Case 5: Missing Address Record (True Positive)
- **S1**: `[S1-851869949] 'Custom Wealth Services LLC' | Addr: 'OH, Columbus, 5559 Orville Avenue...' | US`
- **Target**: `[S3-807524060] 'Custom Wealth Services LLC' | Addr: '...' (Missing Address) | US`
- **Classification**: **TRUE MATCH (Ground Truth Positive)**
- **Generating Blocking Rules**: 5 rules (`exact_name`, `compact_name`, `suffix_name`, `consonant_skel`, `char_ngram`)
- **Key Extracted Features**:
  * `name_exact`: **`1.0000`** | `name_nosuff_exact`: **`1.0000`** | `name_normalized_levenshtein`: **`1.0000`**
  * `candidate_address_missing`: **`1.0000`** | `either_address_missing`: **`1.0000`**
  * `address_exact`: **`0.0000`** | `address_normalized_levenshtein`: **`0.0000`**
  * `num_blocking_rules`: **`5.0000`**
  * *Profile*: `candidate_address_missing == 1.0` informs the model that address metrics are null by omission rather than disagreement, enabling the model to rely on the multi-rule exact name agreement.

### Case 6: Commercial Homonym / Same Name Different City (Distractor / False)
- **S1**: `[S1-865131206] 'Helios' | Addr: '66 Edgewood Street, Bridgeport, CT...' | US`
- **Target**: `[S2-209746679] 'HELIOS' | Addr: 'WINSTON-SALEM, 517 OAK SUMMIT ROAD, NC...' | US`
- **Classification**: **NON-MATCH (Ground Truth Negative)**
- **Generating Blocking Rules**: 4 rules (`exact_name`, `compact_name`, `suffix_name`, `char_ngram`)
- **Key Extracted Features**:
  * `name_exact`: `1.0000` | `name_normalized_levenshtein`: `1.0000`
  * `address_normalized_levenshtein`: **`0.0833`** | `address_token_jaccard`: **`0.0000`**
  * `name_exact_diff_address`: **`1.0000`**
  * *Profile*: While name similarities are perfect ($1.0$), `name_exact_diff_address == 1.0` and `address_token_jaccard == 0.0` provide a distinct negative signal that flags geographic divergence (Bridgeport, CT vs Winston-Salem, NC).

### Case 7: Colocated Building Distractor / Different Business (Distractor / False)
- **S1**: `[S1-629417405] 'Moore Bitwise Inc' | Addr: '337 Oakland Avenue, Michigan City, IN...' | US`
- **Target**: `[S2-471142391] '886 Centgre Realty' | Addr: '#913 OAKLAND AVE, KANSAS CITY, KS...' | US`
- **Classification**: **NON-MATCH (Ground Truth Negative)**
- **Generating Blocking Rules**: 1 rule (`blocked_address_token`: `oakland`)
- **Key Extracted Features**:
  * `name_normalized_levenshtein`: **`0.1111`** | `name_char3_jaccard`: **`0.0333`**
  * `address_normalized_levenshtein`: `0.6571` | `address_token_jaccard`: `0.3333`
  * `num_blocking_rules`: **`1.0000`**
  * *Profile*: Address token triggered the candidate, but name similarity is near zero ($0.1111$) and `num_blocking_rules == 1.0`, enabling clean rejection.

### Case 8: Weak Candidate / Single N-Gram Overlap (Distractor / False)
- **S1**: `[S1-925783039] 'Orelee's Barbershop' | Addr: '1795 Westchester Drive, High Point, NC...' | US`
- **Target**: `[S2-908759962] 'Villarreal Galata Platinum' | Addr: '11129 REGENCY DR, WESTCHESTER, IL...' | US`
- **Classification**: **NON-MATCH (Ground Truth Negative)**
- **Generating Blocking Rules**: 1 rule (`blocked_address_token`: `westchester`)
- **Key Extracted Features**:
  * `name_normalized_levenshtein`: **`0.1538`** | `name_char3_jaccard`: **`0.0000`**
  * `address_token_jaccard`: **`0.2222`** | `house_number_exact`: `0.0000`
  * `num_blocking_rules`: **`1.0000`**
  * *Profile*: Low multi-field agreement and low rule count allow the classifier to filter this pair easily.

---

## 5. Performance, Memory, and Scalability Analysis

| Metric | Validation Cohort (1,000 S1) | Full Validation (10,000 S1) | Full Test Set (50,000 S1) |
|---|---|---|---|
| **Candidate Pairs** | $57,108$ pairs | $\approx 556,858$ pairs | $\approx 2,780,000$ pairs |
| **Feature Extraction Time** | **$11.64$ seconds** | **$\approx 113$ seconds ($1.8$ min)** | **$\approx 560$ seconds ($9.3$ min)** |
| **Throughput** | **$4,904.2$ pairs / sec** | **$\approx 4,900$ pairs / sec** | **$\approx 4,900$ pairs / sec** |
| **Feature Matrix Memory** | **$20.91$ MB** | **$\approx 204$ MB** | **$\approx 1.02$ GB** |
| **Peak Heap RAM** | **$3,081$ MB ($3.01$ GB)** | **$\approx 3,450$ MB ($3.37$ GB)** | **$\approx 4,500$ MB ($4.39$ GB)** |
| **15.5 GB RAM Compliance** | Passed (19.8% utilized) | Passed (22.2% utilized) | Passed (29.0% utilized) |

### Algorithmic Optimizations:
1. **Linear Pairwise Complexity**: Features are evaluated exclusively on candidate pairs ($\mathcal{O}(|\text{Candidates}|) \approx 55 \times N$), completely bypassing the trillions of Cartesian non-candidates.
2. **C++ Native Speed via RapidFuzz**: Levenshtein, token sort, and token set ratios execute compiled C++ routines.
3. **Pre-computed Enriched Records**: Normalization, compact strings, consonant skeletons, and word/n-gram token sets are computed once per entity and shared across all candidate pairs.
4. **Zero Copy Chunk Streaming**: The `extract_batch_vectors` method processes pairs in vectorized chunks of 10,000, avoiding large intermediate Python objects.

---

## 6. Verification and Status

- [x] Implemented `code/business_entity_resolution/src/features.py` computing 96 numeric features across 6 functional groups.
- [x] Implemented explicit address missingness handling (neutral similarity clamping + boolean indicator flags).
- [x] Integrated Configuration H candidate generation and blocking evidence tracking.
- [x] Built comprehensive unit test suite in `code/business_entity_resolution/tests/test_features.py` and `tests/test_features.py`.
- [x] Verified 100% test pass rate (12/12 features tests green, 9/9 blocking tests green, 7/7 preprocessing tests green).
- [x] Executed feature generation benchmark on 57,108 validation candidate pairs (throughput: 4,904.2 pairs/s, 0 NaN, 0 inf, 20.91 MB matrix size).
- [x] Documented complete 96-feature registry with distribution statistics and 8 concrete case studies.
- [x] Strictly adhered to instructions: **No ML models trained, no test predictions generated**.
- [x] Pausing here as instructed.
