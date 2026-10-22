"""source-version-tracker.

Find the claims that need rechecking because the source they cited has changed.
"""

__version__ = "0.1.0"

from .source_versions import *  # noqa: F401,F403
from . import source_versions  # noqa: F401
