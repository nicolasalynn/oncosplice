"""Enable ``python -m oncosplice.weights`` as an alias for the
``oncosplice-download-weights`` console script."""
import sys

from . import _cli

if __name__ == "__main__":
    sys.exit(_cli())
