"""deliverable-checker.

Check a generated deliverable against every requirement in the request that produced it.
"""

__version__ = "0.1.0"

from .deliverables import *  # noqa: F401,F403
from . import deliverables  # noqa: F401
