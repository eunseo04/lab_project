import os

def _get_idle_timeout() -> int:
    """Get the idle timeout from CAI_IDLE_TIMEOUT env var, default 100 seconds."""
    try:
        return int(os.getenv("CAI_IDLE_TIMEOUT", "100"))
    except ValueError:
        return 100


def is_tool_streaming_enabled() -> bool:
    """
    Check if tool output streaming is enabled.

    CAI_TOOL_STREAM controls tool output streaming (default: true)
    CAI_STREAM is ONLY for LLM inference streaming - does NOT affect tools.

    Tools stream by default. Only CAI_TOOL_STREAM=false disables it.
    """
    tool_stream_env = os.getenv("CAI_TOOL_STREAM")
    if tool_stream_env is not None:
        return tool_stream_env.lower() != "false"
    return True  # Default: streaming enabled for tools