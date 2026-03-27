"""Allow running the exploration phase as a module."""

from experiments.generation.exploration.explorer_cli import main

# Import all specs to register them
from experiments.generation.specs import rmsnorm, layernorm, softmax, glu, cross_entropy  # noqa: F401

if __name__ == "__main__":
    import sys
    sys.exit(main())
