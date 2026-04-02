import sys
from pathlib import Path

# Make hyrax_contract_helpers importable from test files in this directory.
# When users copy hyrax_contract_helpers/ into their own test directory they
# should add this same line to their conftest.py.
sys.path.insert(0, str(Path(__file__).parent))
