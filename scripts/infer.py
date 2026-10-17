import sys

from wfr_mfm.cli import main

if __name__ == "__main__":
    main(["infer", *sys.argv[1:]])
