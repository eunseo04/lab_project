"""Research extension; the pre-existing acd package and its results are preserved."""
from pathlib import Path
import acd  # installs the pinned CybORG runtime on the Python path

ROOT = Path(__file__).resolve().parents[1]
