#!/usr/bin/env bash
# ==============================================================================
# ML Challenge 2026: Business Entity Resolution
# AWS Production Inference Orchestration Script (EC2 & Linux Ready)
# ==============================================================================
set -euo pipefail

# 1. Resolve Script Directory & Repository Root Dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

echo "================================================================================"
echo "ML CHALLENGE 2026: AWS EC2 PRODUCTION INFERENCE PIPELINE"
echo "================================================================================"
echo "Repository Root   : $REPO_ROOT"
echo "Current Directory : $(pwd)"
echo "Execution Date    : $(date -u)"
echo "System Hostname   : $(hostname 2>/dev/null || echo 'unknown')"

# 2. Parse Execution Mode & Environment Overrides
MODE="${1:-${MODE:-full}}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/dataset/test}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/code/business_entity_resolution/models}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output}"
BATCH_SIZE="${BATCH_SIZE:-50000}"
THRESHOLD="${THRESHOLD:-0.72}"
FORCE_FRESH="${FORCE_FRESH:-false}"
MAX_S1="${MAX_S1:-}"

# S3 Synchronization URIs (Optional)
S3_DATA_URI="${S3_DATA_URI:-}"
S3_MODEL_URI="${S3_MODEL_URI:-}"
S3_OUTPUT_URI="${S3_OUTPUT_URI:-}"

echo ""
echo "Configuration:"
echo "  - Mode          : $MODE"
echo "  - Data Dir      : $DATA_DIR"
echo "  - Model Dir     : $MODEL_DIR"
echo "  - Output Dir    : $OUTPUT_DIR"
echo "  - Batch Size    : $BATCH_SIZE"
echo "  - Threshold     : $THRESHOLD"
echo "  - Force Fresh   : $FORCE_FRESH"
if [ -n "$S3_DATA_URI" ]; then echo "  - S3 Data URI   : $S3_DATA_URI"; fi
if [ -n "$S3_MODEL_URI" ]; then echo "  - S3 Model URI  : $S3_MODEL_URI"; fi
if [ -n "$S3_OUTPUT_URI" ]; then echo "  - S3 Output URI : $S3_OUTPUT_URI"; fi

# 3. Python & Dependency Verification
echo ""
echo "[1/6] Verifying Python Environment & Installing Dependencies..."
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "ERROR: $PYTHON_BIN not found in PATH!"
    exit 1
fi

REQ_FILE="$REPO_ROOT/code/business_entity_resolution/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    REQ_FILE="$REPO_ROOT/requirements.txt"
fi

if [ -f "$REQ_FILE" ]; then
    echo "Installing requirements from: $REQ_FILE"
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install -r "$REQ_FILE"
else
    echo "WARNING: requirements.txt not found at $REQ_FILE; assuming environment is pre-installed."
fi

# 4. Optional S3 Ingestion (Pre-Inference)
echo ""
echo "[2/6] Verifying Data & Model Assets..."
if [ -n "$S3_DATA_URI" ]; then
    echo "Synchronizing test datasets from S3: $S3_DATA_URI -> $DATA_DIR"
    mkdir -p "$DATA_DIR"
    if ! command -v aws &> /dev/null; then
        echo "ERROR: AWS CLI ('aws') is required for S3 synchronization but was not found in PATH."
        exit 1
    fi
    aws s3 sync "$S3_DATA_URI" "$DATA_DIR"
fi

if [ -n "$S3_MODEL_URI" ]; then
    echo "Synchronizing model artifacts from S3: $S3_MODEL_URI -> $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    if ! command -v aws &> /dev/null; then
        echo "ERROR: AWS CLI ('aws') is required for S3 synchronization but was not found in PATH."
        exit 1
    fi
    aws s3 sync "$S3_MODEL_URI" "$MODEL_DIR"
fi

# Verify Required Files
S1_FILE="$DATA_DIR/test_source1.tsv"
S2_FILE="$DATA_DIR/test_source2.tsv"
S3_FILE="$DATA_DIR/test_source3.tsv"
MODEL_FILE="$MODEL_DIR/best_model.json"
META_FILE="$MODEL_DIR/model_metadata.json"

for f in "$S1_FILE" "$S2_FILE" "$S3_FILE" "$MODEL_FILE" "$META_FILE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

# 5. Build Inference CLI Arguments
INFERENCE_ARGS=(
    "--test-s1-path" "$S1_FILE"
    "--test-s2-path" "$S2_FILE"
    "--test-s3-path" "$S3_FILE"
    "--model-path" "$MODEL_FILE"
    "--metadata-path" "$META_FILE"
    "--output-dir" "$OUTPUT_DIR"
    "--threshold" "$THRESHOLD"
    "--batch-size" "$BATCH_SIZE"
)

if [ "$FORCE_FRESH" = "true" ]; then
    INFERENCE_ARGS+=("--force-fresh")
else
    INFERENCE_ARGS+=("--resume")
fi

if [ "$MODE" = "smoke" ]; then
    SMOKE_LIMIT="${MAX_S1:-10000}"
    echo "Configuring SMOKE mode with max S1 entities: $SMOKE_LIMIT"
    INFERENCE_ARGS+=("--max-s1" "$SMOKE_LIMIT")
elif [ -n "$MAX_S1" ]; then
    INFERENCE_ARGS+=("--max-s1" "$MAX_S1")
fi

# 6. Execute Production Inference Engine
echo ""
echo "[3/6] Launching Python Test Inference Engine..."
INFERENCE_SCRIPT="$REPO_ROOT/code/business_entity_resolution/src/run_full_inference.py"
"$PYTHON_BIN" "$INFERENCE_SCRIPT" "${INFERENCE_ARGS[@]}"

MATCHING_TSV="$OUTPUT_DIR/matching_results.tsv"
CANDIDATE_TSV="$OUTPUT_DIR/candidate_pairs.tsv"

if [ ! -f "$MATCHING_TSV" ] || [ ! -f "$CANDIDATE_TSV" ]; then
    echo "ERROR: Output TSV files were not generated at $OUTPUT_DIR!"
    exit 1
fi

# 7. Run Official Submission Validator
echo ""
echo "[4/6] Running Official Submission Validator..."
VALIDATOR_SCRIPT="$REPO_ROOT/utils/validate_submission.py"
if [ -f "$VALIDATOR_SCRIPT" ]; then
    "$PYTHON_BIN" "$VALIDATOR_SCRIPT" \
        --matching "$MATCHING_TSV" \
        --candidate "$CANDIDATE_TSV" \
        --test-dir "$DATA_DIR"
else
    echo "WARNING: Validator script not found at $VALIDATOR_SCRIPT; skipping."
fi

# 8. Package Submission Zip
echo ""
echo "[5/6] Creating Final Submission Archive..."
SUBMISSION_ZIP="$OUTPUT_DIR/submission.zip"
(
    cd "$OUTPUT_DIR"
    if command -v zip &> /dev/null; then
        zip -9 -q "submission.zip" "matching_results.tsv" "candidate_pairs.tsv"
    else
        echo "INFO: 'zip' utility not found; using python zipfile module."
        "$PYTHON_BIN" -c "
import zipfile, os
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    z.write('matching_results.tsv', arcname='matching_results.tsv')
    z.write('candidate_pairs.tsv', arcname='candidate_pairs.tsv')
"
    fi
)

if [ -f "$SUBMISSION_ZIP" ]; then
    echo "SUCCESS: Final submission archive created: $SUBMISSION_ZIP"
    ls -lh "$SUBMISSION_ZIP"
else
    echo "ERROR: Failed to create submission zip!"
    exit 1
fi

# 9. Optional S3 Upload (Post-Inference)
echo ""
echo "[6/6] Synchronizing Results to S3 (if configured)..."
if [ -n "$S3_OUTPUT_URI" ]; then
    echo "Uploading final output directory to: $S3_OUTPUT_URI"
    if ! command -v aws &> /dev/null; then
        echo "ERROR: AWS CLI ('aws') is required for S3 upload but was not found in PATH."
        exit 1
    fi
    aws s3 sync "$OUTPUT_DIR" "$S3_OUTPUT_URI"
    echo "S3 synchronization complete."
else
    echo "No S3_OUTPUT_URI configured; outputs remain in local $OUTPUT_DIR."
fi

echo ""
echo "================================================================================"
echo "AWS PRODUCTION INFERENCE COMPLETED SUCCESSFULLY"
echo "================================================================================"
