"""Command-line entrypoint for PRISM-DTI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.prepare_stage11b_kiba_binary import *

if __name__ == "__main__":
    main()
