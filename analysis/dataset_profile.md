# Dataset Profile: Business Entity Resolution Challenge

**Generated:** 2026-09-25  
**Evaluation:** Inspection only (No pipeline construction, no file modifications, no external lookups)

---

## 1. Actual File Paths

All files are located within the workspace under `dataset/train/` and `dataset/test/`:

| Dataset Identifier | Relative Path | Absolute Path |
| :--- | :--- | :--- |
| **Train Source 1** | `dataset/train/train_source1.tsv` | `c:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv` |
| **Train Source 2** | `dataset/train/train_source2.tsv` | `c:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source2.tsv` |
| **Train Source 3** | `dataset/train/train_source3.tsv` | `c:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source3.tsv` |
| **Train Ground Truth** | `dataset/train/train_ground_truth.tsv` | `c:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv` |
| **Test Source 1** | `dataset/test/test_source1.tsv` | `c:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source1.tsv` |
| **Test Source 2** | `dataset/test/test_source2.tsv` | `c:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source2.tsv` |
| **Test Source 3** | `dataset/test/test_source3.tsv` | `c:/Users/shaha/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset/test/test_source3.tsv` |

---

## 2. File Sizes

| File | Size (Bytes) | Size (MB) | Size (GB) |
| :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 210,069,713 | 200.34 MB | 0.20 GB |
| `train_source2.tsv` | 489,301,488 | 466.63 MB | 0.47 GB |
| `train_source3.tsv` | 503,705,637 | 480.37 MB | 0.48 GB |
| `train_ground_truth.tsv` | 127,015,583 | 121.13 MB | 0.12 GB |
| **Train Total** | **1,330,092,421** | **1,268.47 MB** | **1.27 GB** |
| `test_source1.tsv` | 175,022,086 | 166.91 MB | 0.17 GB |
| `test_source2.tsv` | 509,456,422 | 485.86 MB | 0.49 GB |
| `test_source3.tsv` | 506,002,772 | 482.56 MB | 0.48 GB |
| **Test Total** | **1,190,481,280** | **1,135.33 MB** | **1.14 GB** |
| **Grand Total (7 files)** | **2,520,573,701** | **2,403.80 MB** | **2.40 GB** |

---

## 3. Row Counts

Exact data row counts (excluding header line, read with `sep="\t"`):

| File | Header Row | Data Rows | Total Records |
| :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 1 | 2,206,821 | 2,206,821 |
| `train_source2.tsv` | 1 | 5,034,616 | 5,034,616 |
| `train_source3.tsv` | 1 | 5,285,603 | 5,285,603 |
| `train_ground_truth.tsv` | 1 | 2,206,821 | 2,206,821 |
| **Train Records Sum** | - | - | **14,733,861** |
| `test_source1.tsv` | 1 | 1,732,544 | 1,732,544 |
| `test_source2.tsv` | 1 | 4,887,273 | 4,887,273 |
| `test_source3.tsv` | 1 | 5,082,316 | 5,082,316 |
| **Test Records Sum** | - | - | **11,702,133** |
| **Combined Records** | - | - | **26,435,994** |

---

## 4. Column Names

Each source file and ground truth file uses tab-separated fields:

| File Type | Columns |
| :--- | :--- |
| **Source Files (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`)** | `entity_id`, `business_name`, `business_address`, `country` |
| **Ground Truth (`train_ground_truth.tsv`)** | `source1_entity_id`, `matched_entity_ids` |

---

## 5. Missing-Value Counts

| File | Total Rows | Missing `entity_id` | Missing `business_name` | Missing `business_address` | Missing `country` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 2,206,821 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `train_source2.tsv` | 5,034,616 | 0 (0.00%) | 0 (0.00%) | 168,967 (3.36%) | 0 (0.00%) |
| `train_source3.tsv` | 5,285,603 | 0 (0.00%) | 0 (0.00%) | 175,916 (3.33%) | 0 (0.00%) |
| `test_source1.tsv` | 1,732,544 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| `test_source2.tsv` | 4,887,273 | 0 (0.00%) | 0 (0.00%) | 129,408 (2.65%) | 0 (0.00%) |
| `test_source3.tsv` | 5,082,316 | 0 (0.00%) | 0 (0.00%) | 136,098 (2.68%) | 0 (0.00%) |

*Observations:*
- `entity_id`, `business_name`, and `country` have zero missing values across all files.
- `business_address` has 0 missing entries in `Source 1` (reference source).
- `business_address` has ~2.65% to 3.36% missing/empty values in `Source 2` and `Source 3`. Matching models must be able to resolve candidates using `business_name` alone when address is blank.

---

## 6. Country Distribution

| File | Total Records | US | India | France |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 2,206,821 | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) |
| `train_source2.tsv` | 5,034,616 | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) |
| `train_source3.tsv` | 5,285,603 | 3,170,056 (59.97%) | 2,115,547 (40.03%) | 0 (0.00%) |
| `test_source1.tsv` | 1,732,544 | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) |
| `test_source2.tsv` | 4,887,273 | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) |
| `test_source3.tsv` | 5,082,316 | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.40%) |

*Key Findings:*
- **Train** contains exclusively `US` (~60%) and `India` (~40%).
- **Test** introduces `France` (~14.4%–15.0%), with `India` (~47%) and `US` (~38%).
- Country proportions across Source 1, Source 2, and Source 3 align within each split.
- Analysis of all 7,638,365 ground truth links reveals **0 cross-country matches**. Country is a 100% hard blocking boundary.

---

## 7. Basic `business_name` and `business_address` Length Statistics

Computed over representative 100,000-row samples per source:

| File | Field | Min Char | Mean Char | Median Char | Max Char | Min Words | Mean Words | Median Words | Max Words |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | `business_name` | 3 | 24.0 | 24.0 | 71 | 1 | 3.5 | 4.0 | 12 |
| | `business_address`| 13 | 52.1 | 41.0 | 222 | 3 | 8.0 | 7.0 | 38 |
| `train_source2.tsv` | `business_name` | 2 | 25.1 | 25.0 | 104 | 1 | 3.5 | 4.0 | 15 |
| | `business_address`| 0 | 46.2 | 37.0 | 202 | 0 | 7.3 | 6.0 | 29 |
| `train_source3.tsv` | `business_name` | 2 | 25.2 | 25.0 | 80 | 1 | 3.5 | 4.0 | 13 |
| | `business_address`| 0 | 46.8 | 42.0 | 198 | 0 | 7.2 | 6.0 | 31 |
| `test_source1.tsv` | `business_name` | 3 | 23.8 | 24.0 | 92 | 1 | 3.5 | 4.0 | 13 |
| | `business_address`| 15 | 57.1 | 50.0 | 216 | 2 | 8.6 | 8.0 | 34 |
| `test_source2.tsv` | `business_name` | 2 | 25.7 | 25.0 | 74 | 1 | 3.6 | 4.0 | 11 |
| | `business_address`| 0 | 50.3 | 42.0 | 226 | 0 | 7.8 | 7.0 | 34 |
| `test_source3.tsv` | `business_name` | 2 | 25.7 | 25.0 | 78 | 1 | 3.6 | 4.0 | 11 |
| | `business_address`| 0 | 48.7 | 43.0 | 213 | 0 | 7.5 | 7.0 | 36 |

---

## 8. Ground-Truth Match-Count Distribution

Analysis of all 2,206,821 Source 1 entities in `train_ground_truth.tsv`:
- **Total ground truth matches**: **7,638,365**
- **Matches from Source 2**: 3,693,619 (48.36%)
- **Matches from Source 3**: 3,944,746 (51.64%)
- **Average matches per S1 entity (all S1)**: 3.46
- **Average matches per non-singleton S1**: 3.66

| Matches per S1 Entity | Count | Percentage of S1 | Cumulative Percentage |
| :--- | :--- | :--- | :--- |
| **0 matches (singletons)** | 123,247 | 5.5848% | 5.58% |
| **1 match** | 119,157 | 5.3995% | 10.98% |
| **2 matches** | 375,212 | 17.0024% | 27.99% |
| **3 matches** | 530,841 | 24.0546% | 52.04% |
| **4 matches** | 484,115 | 21.9372% | 73.98% |
| **5 matches** | 321,957 | 14.5892% | 88.57% |
| **6 matches** | 164,868 | 7.4708% | 96.04% |
| **7 matches** | 63,968 | 2.8986% | 98.94% |
| **8 matches** | 18,680 | 0.8465% | 99.78% |
| **9 matches** | 4,205 | 0.1905% | 99.98% |
| **10 matches** | 534 | 0.0242% | 100.00% |
| **11 matches** | 37 | 0.0017% | 100.00% |

---

## 9. Percentage of Source 1 Entities with Zero, One, and Multiple Matches

| Match Category | S1 Entity Count | Percentage of Total S1 | Evaluation Context ($F_{0.5}$) |
| :--- | :--- | :--- | :--- |
| **Zero Matches (Singletons)** | 123,247 | **5.58%** | Predicting empty returns 1.0; predicting any false match returns 0.0 |
| **Exactly One Match** | 119,157 | **5.40%** | Requires finding the single correct link |
| **Multiple Matches ($\ge 2$)** | 1,964,417 | **89.02%** | Dominant regime; vast majority of entities match 2 to 6 records across S2/S3 |
| **Total S1 Entities** | **2,206,821** | **100.00%** | - |

---

## 10. Estimated $S_1 \times (S_2 + S_3)$ Cartesian Search Space

### 10.1 Global (Unconstrained) Cartesian Space
- **Training Set**:
  $$\text{Train Search Space} = 2,206,821 \times (5,034,616 + 5,285,603) = 2,206,821 \times 10,320,219 = \mathbf{22,774,875,178,801} \approx \mathbf{22.77 \times 10^{12}\ \text{pairs}}$$
- **Test Set**:
  $$\text{Test Search Space} = 1,732,544 \times (4,887,273 + 5,082,316) = 1,732,544 \times 9,969,589 = \mathbf{17,272,750,757,416} \approx \mathbf{17.27 \times 10^{12}\ \text{pairs}}$$

### 10.2 Country-Partitioned Search Space (Leveraging 0 Cross-Country Match Invariant)
Because cross-country matches are strictly 0, the search space decomposes by country:

- **Train by Country**:
  - **US**: $1,323,633 \times (3,016,817 + 3,170,056) = 1,323,633 \times 6,186,873 \approx \mathbf{8.19 \times 10^{12}\ \text{pairs}}$
  - **India**: $883,188 \times (2,017,799 + 2,115,547) = 883,188 \times 4,133,346 \approx \mathbf{3.65 \times 10^{12}\ \text{pairs}}$
  - **Total Train Partitioned**: $\approx \mathbf{11.84 \times 10^{12}\ \text{pairs}}$ (48% reduction vs global)

- **Test by Country**:
  - **US**: $663,106 \times (1,871,330 + 1,945,701) = 663,106 \times 3,817,031 \approx \mathbf{2.53 \times 10^{12}\ \text{pairs}}$
  - **India**: $809,986 \times (2,312,565 + 2,405,000) = 809,986 \times 4,717,565 \approx \mathbf{3.82 \times 10^{12}\ \text{pairs}}$
  - **France**: $259,452 \times (703,378 + 731,615) = 259,452 \times 1,434,993 \approx \mathbf{0.37 \times 10^{12}\ \text{pairs}}$
  - **Total Test Partitioned**: $\approx \mathbf{6.72 \times 10^{12}\ \text{pairs}}$ (61% reduction vs global)

---

## 11. Approximate RAM Requirements & Practical Loading Analysis

### 11.1 System Hardware Constraints
- **Physical RAM**: 15.46 GB (~8–10 GB usable without OS paging)
- **CPU Cores**: 24 cores
- **GPU**: NVIDIA RTX 5070 (8 GB VRAM)

### 11.2 Memory Profiling
1. **Raw On-Disk Size**:
   - Train files: ~1.27 GB
   - Test files: ~1.14 GB
   - Total on-disk footprint: ~2.40 GB
2. **In-Memory Overhead in Standard Pandas (`pd.read_csv`)**:
   - Python objects (`str`) have 48–80 bytes overhead per string instance plus character buffers and pointer arrays.
   - Test set alone consists of **11.70 million rows** $\times$ 4 string columns $\approx$ **46.8 million Python string objects**.
   - Loading all three test files simultaneously into Pandas requires $\approx \mathbf{8.5\text{–}11\text{ GB}}$ of RAM.
   - Loading all four train files simultaneously into Pandas requires $\approx \mathbf{10\text{–}13\text{ GB}}$ of RAM.
   - Loading both train and test simultaneously into Pandas requires $\mathbf{> 20\text{ GB}}$ of RAM.

### 11.3 Feasibility Verdict & Recommended Practical Strategy
- **Full Loading in Standard Pandas: NOT Practical.**
  Simultaneous loading of all train and test tables into standard pandas DataFrames will exceed available system RAM, triggering extreme swap thrashing or an Out-Of-Memory (OOM) crash.
- **Practical, High-Efficiency Strategy**:
  1. **Country Partitioning**: Load and process one country at a time (`France` first $\to$ ~2.4M records across all 3 test files, $\approx$ 1.2 GB RAM; then `US`, then `India`).
  2. **Memory-Efficient Data Structures**: Use PyArrow string arrays, Polars, or compact native Python lists/tuples/generator streams.
  3. **Multi-Stage Candidate Blocking**: Restrict the evaluation space from $6.7 \times 10^{12}$ pairs down to $\approx 10$ candidates per $S_1$ entity ($\approx 17\text{ million}$ candidate pairs for test, $\approx 350\text{ MB}$ TSV), which easily fits into memory for feature extraction and scoring.
