"""isolated_vs_context.py on the v5 split (run through python_v5.sh): test speakers of the v5
protocol, positives = v5 validation positives, near misses from the v5 reference manifest."""
import sys
from pathlib import Path

import isolated_vs_context as iso

iso.RUN_DIR = Path("/home/ml4u/datasets/v5_ref")
iso.POS = Path("logs/v5/positives_v5.jsonl")
sys.exit(iso.main())
