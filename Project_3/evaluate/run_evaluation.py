# -*- coding: utf-8 -*-
"""
Run evaluation script.

This is a convenience script to run the evaluation from the project root.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import evaluation script
from scripts.evaluate_models import main

if __name__ == "__main__":
    # Run evaluation
    main()