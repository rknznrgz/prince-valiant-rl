"""Paths anchored to this source checkout, independent of the working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'config'
