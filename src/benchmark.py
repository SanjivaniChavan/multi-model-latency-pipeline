import statistics
import time

from .pipeline import chat, ChatRequest
from .utils import log


def benchmark(n_runs: int = 10):
    latencies = []

    for i in range(n_runs):
        start = time.perf_counter()
        _ = chat(ChatRequest(text="Explain why low-latency AI pipelines matter."))
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000.0
        latencies.append(elapsed_ms)
        log(f"Run {i+1}/{n_runs}: {elapsed_ms:.2f} ms")

    avg = statistics.mean(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)

    print("\n========== BENCHMARK RESULTS ==========")
    print(f"Average latency: {avg:.2f} ms")
    print(f"P95 latency:    {p95:.2f} ms")
    print("=======================================\n")


if __name__ == "__main__":
    benchmark()
