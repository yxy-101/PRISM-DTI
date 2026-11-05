"""Command-line entrypoint for PRISM-DTI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.clean_biosnap_pair_leakage import *

if __name__ == "__main__":
    main()
