"""Command-line entrypoint for PRISM-DTI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.prepare_stage8c_davis_binary_labels import *

if __name__ == "__main__":
    main()
