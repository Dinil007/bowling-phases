"""Deprecated alias kept for compatibility. Use: python -m src.run --manifest <csv> --out <dir>"""
import sys
from src.run import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and not args[0].startswith("-"):
        args = ["--manifest"] + args
    sys.exit(main(args))
