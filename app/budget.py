"""BudgetGuard — the real bound on execution, independent of any framework.

Checked at the entry of every node. A breach raises ``BudgetExceeded``; the
runner catches it, records the reason, and jumps straight to ``compose_response``
so the caller always gets a structured (possibly partial) answer — never an
exception and never an unbounded loop.
"""

from __future__ import annotations

import time

from app.state import BudgetState

# Which budget dimension each node can push against.
_LLM_NODES = {"interpret_mandate", "validate_and_rank"}
_ITER_NODES = {"retrieve"}
_REVISION_NODES = {"relax_preferences"}


class BudgetExceeded(RuntimeError):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(detail or kind)
        self.kind = kind          # "deadline" | "llm_calls" | "iterations" | "revisions"


class BudgetGuard:
    def __init__(self, budget: BudgetState, deadline_ts: float | None = None,
                 *, clock=time.monotonic):
        self.budget = budget
        self.deadline_ts = deadline_ts
        self._clock = clock

    def time_left(self) -> float | None:
        if self.deadline_ts is None:
            return None
        return self.deadline_ts - self._clock()

    def before(self, node: str) -> None:
        """Raise if running ``node`` now would break a bound."""
        b = self.budget

        if self.deadline_ts is not None and self._clock() >= self.deadline_ts:
            raise BudgetExceeded("deadline", f"wall-clock deadline reached before {node}")

        if node in _LLM_NODES and b.llm_calls >= b.max_llm_calls:
            raise BudgetExceeded(
                "llm_calls", f"{b.llm_calls}/{b.max_llm_calls} LLM calls used before {node}"
            )
        if node in _ITER_NODES and b.iterations >= b.max_iterations:
            raise BudgetExceeded(
                "iterations", f"{b.iterations}/{b.max_iterations} retrieval iterations used"
            )
        if node in _REVISION_NODES and b.revisions >= b.max_revisions:
            raise BudgetExceeded(
                "revisions", f"{b.revisions}/{b.max_revisions} revisions used"
            )

    def clamp_config(self, cfg) -> None:
        """Pull runtime knobs down under the hard ceilings (a caller can only ever
        ask for *less* than the budget allows)."""
        b = self.budget
        cfg.pool_limit = min(cfg.pool_limit, b.max_pool)
        cfg.validation_batch = min(cfg.validation_batch, b.max_validation_batch)
        cfg.result_limit = min(cfg.result_limit, b.max_results)
