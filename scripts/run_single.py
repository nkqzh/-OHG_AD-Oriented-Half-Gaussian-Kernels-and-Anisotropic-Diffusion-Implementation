# --- bootstrap: allow running without install ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------------------------
from codes.runners.single import main_cli

if __name__ == "__main__":
    main_cli()
