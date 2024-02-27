import os
import argparse
import numpy as np
from utils import setup_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)


    args = parser.parse_args()
    setup_seed(args.seed)
    
    