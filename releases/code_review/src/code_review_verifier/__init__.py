"""code-review-verifier.

Check whether an AI review comment is technically true of the code it points at.
"""

__version__ = "0.1.0"

from .code_review import *  # noqa: F401,F403
from . import code_review  # noqa: F401
