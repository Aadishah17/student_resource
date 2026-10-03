"""
Create/Update a submission snapshot from the current committed production output.
1. Reads committed byte offsets from output/.inference_checkpoint.json.
2. Writes committed records to output/submission_snapshot_now/matching_results.tsv
   and output/submission_snapshot_now/candidate_pairs.tsv.
3. Reads all 1,732,544 S1 IDs from dataset/test/test_source1.tsv.
4. Appends all unprocessed S1 IDs with empty match and empty candidate entries (<s1_id>\\t\\n).
5. Asserts exact row counts, 1-to-1 matching, and no duplicates.
6. Packages into output/submission_snapshot_now/submission.zip.
"""
import os
import sys
import json
import zipfile
import time
import subprocess

def create_snapshot():
    checkpoint_path = os.path.join("output", ".inference_checkpoint.json")
    src_matching_path = os.path.join("output", "matching_results.tsv")
    src_candidate_path = os.path.join("output", "candidate_pairs.tsv")
    s1_dataset_path = os.path.join("dataset", "test", "test_source1.tsv")
    
    target_dir = os.path.join("output", "submission_snapshot_now")
    os.makedirs(target_dir, exist_ok=True)
    
    tgt_matching_path = os.path.join(target_dir, "matching_results.tsv")
    tgt_candidate_path = os.path.join(target_dir, "candidate_pairs.tsv")
    zip_path = os.path.join(target_dir, "submission.zip")
    
    print("=" * 70)
    print("SUBMISSION SNAPSHOT GENERATION")
    print("=" * 70)
    
    # 1. Read checkpoint state
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
        
    matching_offset = ckpt["matching_byte_offset"]
    candidate_offset = ckpt["candidate_byte_offset"]
    written_count = ckpt["written_s1_count"]
    last_batch = ckpt.get("last_batch", "unknown")
    completed_batches = ckpt.get("completed_batches", [])
    
    print(f"Checkpoint Information:")
    print(f"  - Completed Batches: {len(completed_batches)}")
    print(f"  - Last Batch       : {last_batch}")
    print(f"  - Written S1 Count : {written_count:,}")
    print(f"  - Matching Offset  : {matching_offset:,} bytes")
    print(f"  - Candidate Offset : {candidate_offset:,} bytes")
    
    # 2. Copy committed matching results and track seen S1 IDs
    print(f"\n[1/5] Copying committed matching results to {tgt_matching_path}...")
    seen_s1 = set()
    matching_committed_count = 0
    
    with open(src_matching_path, "rb") as fin, open(tgt_matching_path, "wb") as fout:
        data = fin.read(matching_offset)
        if len(data) != matching_offset:
            raise ValueError(f"Expected {matching_offset} bytes from {src_matching_path}, got {len(data)}")
        fout.write(data)
        
    with open(tgt_matching_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        assert header == "source1_entity_id\tmatched_entity_ids", f"Unexpected header: {header}"
        for line in f:
            s1_id = line.split("\t", 1)[0].strip()
            if s1_id:
                seen_s1.add(s1_id)
                matching_committed_count += 1
                
    print(f"  Committed matching rows copied: {matching_committed_count:,}")
    assert matching_committed_count == written_count, (
        f"Row count mismatch: {matching_committed_count} vs checkpoint {written_count}"
    )
    
    # 3. Copy committed candidate pairs and verify alignment
    print(f"\n[2/5] Copying committed candidate pairs to {tgt_candidate_path}...")
    candidate_committed_count = 0
    cand_seen_s1 = set()
    
    with open(src_candidate_path, "rb") as fin, open(tgt_candidate_path, "wb") as fout:
        data = fin.read(candidate_offset)
        if len(data) != candidate_offset:
            raise ValueError(f"Expected {candidate_offset} bytes from {src_candidate_path}, got {len(data)}")
        fout.write(data)
        
    with open(tgt_candidate_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        assert header == "source1_entity_id\tcandidate_entity_ids", f"Unexpected header: {header}"
        for line in f:
            s1_id = line.split("\t", 1)[0].strip()
            if s1_id:
                cand_seen_s1.add(s1_id)
                candidate_committed_count += 1
                
    print(f"  Committed candidate rows copied: {candidate_committed_count:,}")
    assert cand_seen_s1 == seen_s1, "Mismatch between matching and candidate seen S1 IDs!"
    
    # 4. Read all test S1 IDs and append missing rows
    print(f"\n[3/5] Ingesting all test S1 IDs from {s1_dataset_path}...")
    all_s1_ids = []
    with open(s1_dataset_path, "r", encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.split("\t", 1)
            if parts and parts[0].strip():
                all_s1_ids.append(parts[0].strip())
                
    total_test_s1 = len(all_s1_ids)
    print(f"  Total test S1 entities: {total_test_s1:,}")
    assert total_test_s1 == 1732544, f"Expected 1,732,544 rows, found {total_test_s1}"
    
    missing_s1_ids = [eid for eid in all_s1_ids if eid not in seen_s1]
    missing_count = len(missing_s1_ids)
    print(f"  Existing completed S1: {len(seen_s1):,} ({len(seen_s1)/total_test_s1*100:.2f}%)")
    print(f"  Missing S1 to complete: {missing_count:,} ({missing_count/total_test_s1*100:.2f}%)")
    
    # Append missing rows (<eid>\t\n)
    print(f"  Appending {missing_count:,} missing rows to matching and candidate TSVs...")
    with open(tgt_matching_path, "a", encoding="utf-8", newline="") as fm, \
         open(tgt_candidate_path, "a", encoding="utf-8", newline="") as fc:
        for eid in missing_s1_ids:
            fm.write(f"{eid}\t\n")
            fc.write(f"{eid}\t\n")
            
    # Verify final line counts
    def count_lines(path):
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
            
    m_lines = count_lines(tgt_matching_path)
    c_lines = count_lines(tgt_candidate_path)
    print(f"  Final {tgt_matching_path}: {m_lines:,} lines (1 header + {m_lines - 1:,} rows)")
    print(f"  Final {tgt_candidate_path}: {c_lines:,} lines (1 header + {c_lines - 1:,} rows)")
    assert m_lines == 1732545, f"Expected 1,732,545 lines, got {m_lines}"
    assert c_lines == 1732545, f"Expected 1,732,545 lines, got {c_lines}"
    
    # 5. Run official submission validation
    print(f"\n[4/5] Running official validate_submission.py...")
    val_cmd = [
        sys.executable,
        os.path.join("utils", "validate_submission.py"),
        "--matching", tgt_matching_path,
        "--candidate", "none",
        "--test-dir", os.path.join("dataset", "test")
    ]
    val_res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(val_res.stdout)
    if val_res.returncode != 0:
        print(val_res.stderr)
        raise RuntimeError(f"validate_submission.py failed with code {val_res.returncode}")
        
    # 6. Create submission ZIP archive
    print(f"\n[5/5] Creating submission archive at {zip_path}...")
    t_zip_start = time.time()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        print("  Adding matching_results.tsv to zip...")
        z.write(tgt_matching_path, arcname="matching_results.tsv")
        print("  Adding candidate_pairs.tsv to zip...")
        z.write(tgt_candidate_path, arcname="candidate_pairs.tsv")
        
    zip_size = os.path.getsize(zip_path)
    print(f"  Submission ZIP created in {time.time()-t_zip_start:.2f}s: {zip_size:,} bytes ({zip_size/(1024**2):.2f} MB)")
    
    print("\n" + "=" * 70)
    print("SUBMISSION SNAPSHOT UPDATE SUCCESSFUL")
    print(f"Updated matching_results: {tgt_matching_path}")
    print(f"Total Completed S1 predictions: {written_count:,} ({written_count/1732544*100:.2f}%)")
    print(f"Total Rows in File: 1,732,544 S1 rows")
    print("=" * 70)

if __name__ == "__main__":
    create_snapshot()
