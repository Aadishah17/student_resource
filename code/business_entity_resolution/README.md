# Business Entity Resolution Pipeline

This folder contains the reusable preprocessing component for the ML Challenge
2026 business entity resolution task.

## Layout

- `src/preprocessing.py` — name, address, transliteration, and record-level preprocessing
- `src/demo_preprocessing.py` — runnable examples using representative records
- `tests/test_preprocessing.py` — unit and integration tests
- `requirements.txt` — runtime dependency constraints

## Setup

From this directory, install the runtime dependency:

```bash
python -m pip install -r requirements.txt
```

## Run the tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Run the demo

```bash
python src/demo_preprocessing.py
```

The demo uses only in-memory sample records. The preprocessing module can also
process challenge TSV files in chunks with `stream_tsv_preprocessed`.
