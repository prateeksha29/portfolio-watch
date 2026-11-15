"""Private add-ons: modules that exist only in the owner's copy (git-ignored), such as a newsletter read with the
owner's own subscription. The public code runs without them: optional() returns None when the module isn't there."""
import importlib
from types import ModuleType


def optional(name: str) -> ModuleType | None:
    """fin_news.<name>, or None if that file doesn't exist (an error inside an add-on that does exist still raises)."""
    full = f"{__package__}.{name}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as e:
        if e.name != full:
            raise
        return None


advisor = optional("advisor")  # a newsletter's picks and ratings; its web pages are in advisor_web.py
