"""Domain-specific exceptions with user-actionable messages."""


class ContractError(ValueError):
    """Raised when research data violates a declared contract."""


class FoldError(ValueError):
    """Raised when leakage-safe balanced folds cannot be constructed."""


class WindowError(ValueError):
    """Raised when deterministic temporal windows cannot be constructed."""

