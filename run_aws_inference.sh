#!/usr/bin/env bash
# ==============================================================================
# ML Challenge 2026: Business Entity Resolution
# AWS Full-Scale Test Inference Execution Script
# ==============================================================================
set -e

echo "================================================================================"
echo "STARTING FULL TEST INFERENCE PIPELINE ON AWS"
echo "================================================================================"
echo "Current directory: $(pwd)"
echo "Date: $(date -u)"

# 1. Verify Python & Dependencies
echo ""
echo "[1/4] Setting up Python environment & dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 2. Run Full Test Inference Engine
echo ""
echo "[2/4] Running country-partitioned full test inference (1.73M S1 entities)..."
python3 code/business_entity_resolution/src/run_full_inference.py \
    --batch-size 50000 \
    --matching-out output/matching_results.tsv \
    --candidate-out output/candidate_pairs.tsv

# 3. Run Official Validator
echo ""
echo "[3/4] Running official submission validator..."
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 4. Package Submission Zip
echo ""
echo "[4/4] Creating final submission zip archive..."
if [ -f output/matching_results.tsv ] && [ -f output/candidate_pairs.tsv ]; then
    cd output
    zip -9 submission.zip matching_results.tsv candidate_pairs.tsv
    cd ..
    echo "SUCCESS: Final submission archive created at: output/submission.zip"
    ls -lh output/submission.zip
else
    echo "ERROR: Output TSV files not found! Cannot create submission zip."
    exit 1
fi

echo "================================================================================"
echo "FULL AWS TEST INFERENCE COMPLETED SUCCESSFULLY"
echo "================================================================================"
