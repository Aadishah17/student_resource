import os
import sys
import time
import multiprocessing as mp
import numpy as np

def worker_task(worker_id, start_idx, end_idx, raw_arr_shape, raw_arr):
    # Map raw array to numpy view
    arr = np.frombuffer(raw_arr, dtype=np.float32).reshape(raw_arr_shape)
    for i in range(start_idx, end_idx):
        arr[i, :] = worker_id + 0.5

def main():
    n_rows = 100000
    n_cols = 96
    num_workers = 4
    
    raw_arr = mp.RawArray('f', n_rows * n_cols)
    chunk_size = (n_rows + num_workers - 1) // num_workers
    
    procs = []
    for w in range(num_workers):
        start = w * chunk_size
        end = min(start + chunk_size, n_rows)
        if start >= end:
            continue
        p = mp.Process(target=worker_task, args=(w, start, end, (n_rows, n_cols), raw_arr))
        p.start()
        procs.append(p)
        
    for p in procs:
        p.join()
        
    res = np.frombuffer(raw_arr, dtype=np.float32).reshape((n_rows, n_cols))
    print(f"Result shape: {res.shape}")
    print(f"Row 0 (Worker 0): {res[0, :3]}")
    print(f"Row {chunk_size} (Worker 1): {res[chunk_size, :3]}")
    print(f"Verification: all nonzero? {np.all(res > 0)}")

if __name__ == '__main__':
    main()
