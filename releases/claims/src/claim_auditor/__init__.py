"""claim-auditor.

Check whether the claims in an AI answer are supported by the sources it was given.
"""

__version__ = "0.1.0"

from .claims import *  # noqa: F401,F403
from . import claims  # noqa: F401
