"""Run the official Image entry with a 50 decimal MB complete-file budget."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts._common.io import ROOT
from scripts._common.run import main

if __name__ == '__main__':
    main('ingp','gigapixel',ROOT/'configs/baselines/ingp_gigapixel_50mib.json')
