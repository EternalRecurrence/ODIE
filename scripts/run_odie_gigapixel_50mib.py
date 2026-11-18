"""Launch ODIE through the single shared training implementation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts._common.odie import main
from scripts._common.io import ROOT

if __name__=='__main__':
    main('gigapixel',ROOT/'configs/baselines/odie_gigapixel_50mib.json')
