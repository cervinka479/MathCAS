import contextlib
import os
from typing import Optional

try:
    import nvtx
except ImportError:
    nvtx = None

# Global flag to enable/disable NVTX profiling
# Can be controlled via environment variable or set programmatically
ENABLE_NVTX = os.getenv('ENABLE_NVTX', 'false').lower() in ('true', '1', 'yes')

@contextlib.contextmanager
def nvtx_range(msg: str, color: Optional[str] = None):
    """
    Context manager for NVTX range profiling.
    
    Args:
        msg: Message/name for the profiling range
        color: Optional color for the range (e.g., 'red', 'blue', 'green')
    """
    if ENABLE_NVTX and nvtx:
        if color:
            range_id = nvtx.start_range(message=msg, color=color)
        else:
            range_id = nvtx.start_range(message=msg)
        try:
            yield
        finally:
            nvtx.end_range(range_id)
    else:
        yield

def enable_profiling(enable: bool = True):
    """Enable or disable NVTX profiling globally."""
    global ENABLE_NVTX
    ENABLE_NVTX = enable

def is_profiling_enabled() -> bool:
    """Check if NVTX profiling is currently enabled."""
    return ENABLE_NVTX and nvtx is not None

def nvtx_mark(msg: str, color: Optional[str] = None):
    """Add an instant marker to the timeline."""
    if ENABLE_NVTX and nvtx:
        if color:
            nvtx.mark(message=msg, color=color)
        else:
            nvtx.mark(message=msg)