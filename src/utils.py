import time
from contextlib import contextmanager
from typing import Callable, Any, Dict

from rich import print


def log(msg: str):
    """Simple rich-based logger."""
    print(f"[bold cyan][PIPELINE][/bold cyan] {msg}")


@contextmanager
def timed(section: str) -> Dict[str, float]:
    """
    Context manager to measure elapsed time for a code block.
    Usage:
        with timed("ASR") as t:
            ...
        print(t["elapsed_ms"])
    """
    start = time.perf_counter()
    timings: Dict[str, float] = {}
    try:
        yield timings
    finally:
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0
        timings["elapsed_ms"] = elapsed_ms
        print(f"[bold green]{section}[/bold green] took {elapsed_ms:.2f} ms")


def safe_call(name: str, fn: Callable, *args, **kwargs) -> Any:
    """
    Helper to call a model component and log errors without crashing the pipeline.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[bold red][ERROR][/bold red] in {name}: {exc}")
        return None
