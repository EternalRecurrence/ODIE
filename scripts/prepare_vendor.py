"""Prepare pinned upstream checkouts and apply ODIE patches on Linux."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts._common.vendor import main

if __name__=='__main__':
    main()
