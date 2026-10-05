from typing import TYPE_CHECKING

class AgentsException(Exception):
    """Base class for all exceptions in the CAI Agents."""


class MaxTurnsExceeded(AgentsException):
    """Exception raised when the maximum number of turns is exceeded."""

    message: str

    def __init__(self, message: str):
        self.message = message


class ModelBehaviorError(AgentsException):
    """Exception raised when the model does something unexpected, e.g. calling a tool that doesn't
    exist, or providing malformed JSON.
    """

    message: str

    def __init__(self, message: str):
        self.message = message


class UserError(AgentsException):
    """Exception raised when the user makes an error using CAI."""

    message: str

    def __init__(self, message: str):
        self.message = message

class PriceLimitExceeded(AgentsException):
    """Raised when the maximum price limit is exceeded."""
    def __init__(self, current_cost: float, price_limit: float):
        super().__init__(f"Maximum price limit (${price_limit:.4f}) exceeded. Current cost: ${current_cost:.4f}")