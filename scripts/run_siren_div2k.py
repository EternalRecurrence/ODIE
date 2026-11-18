"""Launch the pinned official siren image training entry."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._common.run import main

if __name__ == '__main__':
    main('siren', 'div2k')
