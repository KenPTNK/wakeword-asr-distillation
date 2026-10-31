"""dump_scores.py on the v5 split: test speakers of v5/proto/incontext_protocol.py (run through
python_v5.sh), positives = validation positives of the v5 manifest, near misses from the v5
reference unified manifest.  Same scoring, seeds and hk15 stream as the frozen protocol."""
import sys
from pathlib import Path

import dump_scores as ds

ds.RUN_DIR = Path("/home/ml4u/datasets/v5_ref")
ds.POS = Path("logs/v5/positives_v5.jsonl")
sys.exit(ds.main())
