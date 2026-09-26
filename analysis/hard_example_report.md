# Hard-Example Enrichment Experiment & Error Analysis Report

> **Stage 8: Hard-Example Mining, Controlled Enrichment, and Boundary Calibration**  
> **Repository:** `ML Challenge 2026: Business Entity Resolution`  
> **Training Base:** 50,000 Source 1 Entities (780,656 Candidate Pairs)  
> **Evaluation Base:** Strict Held-Out 2,000 S1 Entity Validation Cohort (106,489 Candidate Pairs)  
> **Decision Rule:** Strict Validation Macro $F_{0.5}$ Maximization  
> **Retained Champion:** **XGBoost (Baseline 50k Cohort, Threshold $\tau = 0.72$)**  
> **Status:** Completed & Empirically Verified

---

## Executive Summary

To explore whether under-represented difficult edge cases (such as missing target addresses, Indic transliteration divergence, commercial homonyms, and acronyms) could be mitigated via sample re-balancing, we conducted a systematic hard-example enrichment experiment using the verified 50,000-entity training dataset.

### Core Comparison Matrix

| Configuration | Description | Training Pairs | Validation Macro $F_{0.5}$ | Pairwise Precision | Pairwise Recall | Singleton Accuracy | Optimal Threshold | False Merges | False Negatives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** | **Baseline 50k Natural Distribution** | **780,656** | **97.48%** | 98.69% | **97.53%** | 99.00% | **0.72** | 88 | **168** |
| **Config B** | **50k + Moderate Hard Positives** | 861,589 | **97.35%** | 98.83% | 97.06% | 99.00% | 0.84 | 78 | 200 |
| **Config C** | **50k + Hard Positives + Hard Negatives** | 934,350 | **97.26%** | **98.94%** | 96.36% | **100.00%** | 0.82 | **70** | 248 |

---

## 1. Hard Example Identification & Distribution Census

Using the forensic findings from the Stage 6 error analysis, we constructed reproducible rule filters over the 96 features in `train_data_50k.npz`:

### A. Difficult Positive Matches (74,260 pairs, 44.18% of positives)
- **Missing Candidate Address (6,821 pairs):** Target record lacks street address text; model tends to penalize missingness.
- **Physical Address Divergence (9,109 pairs):** Genuine corporate branch vs headquarters address mismatch (`address_token_jaccard < 0.35`).
- **Low Lexical Name Similarity (60,287 pairs):** Heavy character corruptions, acronyms, or trade names (`name_normalized_levenshtein < 0.65`).
- **Cross-Script Latin $\leftrightarrow$ Indic (10,869 pairs):** Script transitions requiring phonological bridging.
- **Transliteration Divergence (3,412 pairs):** Non-standard Romanization variants.
- **Missing Postal Code & House Number (22,410 pairs):** Sparse regional records without numeric anchors.

### B. Difficult Negative Distractors (72,761 pairs, 11.88% of negatives)
- **Commercial Homonyms & Franchises (24,269 pairs):** Exact business name operating at divergent street addresses (`name_exact_diff_address == 1.0`).
- **High Name Match + Missing Address (1,194 pairs):** High name similarity without address confirmation to refute the match.
- **Same House / Postal Distractors (279 pairs):** Co-located distinct businesses sharing building or postal codes.
- **Cross-Script Distractors (506 pairs):** Accidental phonological collisions across scripts.
- **Multi-Rule Agreement Distractors (48,507 pairs):** Spurious candidates triggering $\ge 3$ blocking rules despite physical divergence.

---

## 2. Experimental Results & Theoretical Root-Cause Analysis

### Why Did Hard-Example Enrichment Not Improve Validation Macro $F_{0.5}$?

1. **Prior Probability Distortion & Decision Threshold Drift:**
   - In Config A, the natural blocking candidate distribution produces an optimal threshold of **$\tau = 0.72$**, achieving an exceptional balance between precision (**98.69%**) and recall (**97.53%**).
   - In Config B, artificially duplicating hard positives inflated the positive prior on ambiguous candidates. The tree responded by elevating prediction probabilities across borderline non-matches, forcing the optimal threshold to jump to **$\tau = 0.84$** to suppress false merges.
   - At $\tau = 0.84$, the model rejected marginal true matches, reducing pairwise recall from **97.53% down to 97.06%** and increasing false negatives from 168 to 200.

2. **Precision–Recall Trade-Off under the Competition Metric:**
   - In Config C, duplicating hard negatives produced near-perfect false-positive avoidance: precision reached **98.94%**, and singleton accuracy reached **100.00%** (zero false merges on singletons).
   - However, this extreme conservatism caused pairwise recall to plummet by **-1.17%** (from 97.53% to 96.36%), driving false negatives up to 248.
   - Because Macro $F_{0.5}$ weights precision more than recall ($F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$), the +0.25% gain in precision was insufficient to overcome the -1.17% loss in recall, dropping Macro $F_{0.5}$ from **97.48% down to 97.26%**.

3. **Natural Blocking Negative Quality:**
   - Configuration H blocking already performs hard-negative mining by construction. The candidates evaluated by the tree are not random Cartesian negatives; they are already tightly clustered distractors sharing exact tokens, skeletons, and postal codes.
   - Further artificial replication of these pairs causes overfitting on the re-sampled distractor modes without improving general boundary discernment.

---

## 3. Error Analysis on the Best Enriched Model (Config B)

Comparing the prediction error distributions between Config A (Baseline 50k) and Config B (Enriched):

```
Error Metric                  Config A (Baseline)    Config B (Enriched)    Impact of Enrichment
------------------------------------------------------------------------------------------------
Pairwise False Merges (FP)    88                     78                     -10 False Positives
Pairwise Missed Links (FN)    168                    200                    +32 False Negatives
Singleton Accuracy            99.00%                 99.00%                 No change (1/100 error)
Optimal Threshold             0.72                   0.84                   +0.12 shift
Validation Macro F0.5         97.48%                 97.35%                 -0.13% Degradation
```

### Representative Failure Case Study (Config B Misses):
- **Source 1 Record:** `"Vardhman Textiles Limited"` | `"Chandigarh Road, Ludhiana, Punjab 141010"`
- **Target Candidate (S3):** `"Vardhman Textiles Ltd"` | `""` (Missing Address)
- **Config A Probability:** $p = 0.742 \ge 0.72$ $\to$ **CORRECT MATCH (True Positive)**
- **Config B Probability:** $p = 0.781 < 0.84$ $\to$ **MISSED MATCH (False Negative)**
- **Forensic Diagnosis:** Although the model's assigned probability on this missing-address pair rose from $0.742$ to $0.781$, the global decision threshold was forced to shift even higher ($0.84$) due to elevated false positive risk elsewhere. Consequently, valid true matches that succeeded under Config A were rejected under Config B.

---

## 4. Final Retention Decision

Per the explicit evaluation criteria:
> *"If enrichment does not improve Macro F0.5, retain the current 50k model."*

Because both Config B (97.35%) and Config C (97.26%) underperformed the natural distribution baseline (**97.48%**), the **unmodified 50k XGBoost model at threshold $\tau = 0.72$ is definitively retained as the final champion**.

---

## 5. Final Audit Summary

```
================================================================================
FINAL HARD-EXAMPLE ENRICHMENT AUDIT
================================================================================
1. baseline 50k F0.5           : 97.48%
2. enriched model F0.5         : 97.35% (Config B)
3. improvement/difference      : -0.13%
4. selected threshold          : 0.72
5. precision                   : 98.69%
6. recall                      : 97.53%
7. singleton accuracy          : 99.00%
8. false merges                : 88
9. false negatives             : 168
10. training size              : 780,656 pairs (Retained Champion)
11. runtime                    : ~35 seconds
12. RAM                        : ~750 MB
================================================================================
```
