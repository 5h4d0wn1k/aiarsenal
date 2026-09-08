"""Allow `python -m aiarsenal` to behave like the console script."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())