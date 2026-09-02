import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported seamlessly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui.app import main

if __name__ == "__main__":
    main()
