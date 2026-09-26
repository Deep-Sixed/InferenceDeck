"""Portable core for InferenceDeck.

This package intentionally avoids machine-specific defaults. Discovery comes
from environment variables, PATH, standard per-OS app/cache locations, and the
current project root.
"""

from .inventory import build_inventory
from .profile_resolver import resolve_profiles

__all__ = ["build_inventory", "resolve_profiles"]
__version__ = "0.1.0"
__license__ = "MIT"

from .hf_cli import detect_hf_cli, check_for_updates, install_hf_cli
