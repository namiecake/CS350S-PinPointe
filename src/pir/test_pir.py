"""
Tests for PIR implementation.

Run with: python -m pir.test_pir
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pir.pir_scheme import test_pir_basic

if __name__ == "__main__":
    test_pir_basic()
    print("\n✓ All PIR tests passed!")