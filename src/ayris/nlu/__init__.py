"""Natural language understanding: matching, slot extraction, context, LLM.

Turns recognised speech into a resolved command with arguments, or hands the
utterance over to an LLM depending on the configured NLU mode.
"""

from __future__ import annotations
