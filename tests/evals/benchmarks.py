"""
Benchmark suite for AI Terminal Agent performance testing.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import time
import statistics
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float  # operations per second
    memory_usage_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteResult:
    """Result of a benchmark suite."""
    suite_name: str
    timestamp: datetime
    total_benchmarks: int
    total_time: float
    results: List[BenchmarkResult] = field(default_factory=list)


class Benchmark:
    """
    Benchmark runner for performance testing.
    """

    def __init__(self, name: str, warmup_iterations: int = 3):
        self.name = name
        self.warmup_iterations = warmup_iterations
        self.times: List[float] = []

    async def run(self, func: Callable, iterations: int = 100,
                 *args, **kwargs) -> BenchmarkResult:
        """
        Run a benchmark.

        Args:
            func: Function to benchmark
            iterations: Number of iterations
            *args, **kwargs: Arguments to pass to function

        Returns:
            BenchmarkResult
        """
        self.times = []

        # Warmup
        for _ in range(self.warmup_iterations):
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)

        # Actual benchmark
        for _ in range(iterations):
            start = time.perf_counter()

            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)

            elapsed = time.perf_counter() - start
            self.times.append(elapsed)

        # Calculate statistics
        total_time = sum(self.times)
        avg_time = statistics.mean(self.times)
        min_time = min(self.times)
        max_time = max(self.times)
        std_dev = statistics.stdev(self.times) if len(self.times) > 1 else 0

        return BenchmarkResult(
            name=self.name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            throughput=iterations / total_time if total_time > 0 else 0
        )


class BenchmarkSuite:
    """
    Suite of benchmarks.
    """

    def __init__(self, name: str):
        self.name = name
        self.benchmarks: List[tuple] = []

    def add(self, name: str, func: Callable, iterations: int = 100,
            *args, **kwargs) -> None:
        """Add a benchmark to the suite."""
        self.benchmarks.append((name, func, iterations, args, kwargs))

    async def run_all(self) -> BenchmarkSuiteResult:
        """Run all benchmarks in the suite."""
        start_time = time.perf_counter()
        results = []

        for name, func, iterations, args, kwargs in self.benchmarks:
            print(f"Running benchmark: {name}...")
            benchmark = Benchmark(name)
            result = await benchmark.run(func, iterations, *args, **kwargs)
            results.append(result)
            print(f"  Avg: {result.avg_time*1000:.2f}ms, "
                  f"Throughput: {result.throughput:.2f} ops/s")

        total_time = time.perf_counter() - start_time

        return BenchmarkSuiteResult(
            suite_name=self.name,
            timestamp=datetime.now(),
            total_benchmarks=len(results),
            total_time=total_time,
            results=results
        )


class BenchmarkReporter:
    """
    Generates benchmark reports.
    """

    @staticmethod
    def to_json(result: BenchmarkSuiteResult) -> str:
        """Convert results to JSON."""
        data = {
            "suite_name": result.suite_name,
            "timestamp": result.timestamp.isoformat(),
            "total_benchmarks": result.total_benchmarks,
            "total_time": result.total_time,
            "benchmarks": []
        }

        for r in result.results:
            data["benchmarks"].append({
                "name": r.name,
                "iterations": r.iterations,
                "avg_time_ms": r.avg_time * 1000,
                "min_time_ms": r.min_time * 1000,
                "max_time_ms": r.max_time * 1000,
                "std_dev_ms": r.std_dev * 1000,
                "throughput_ops_s": r.throughput
            })

        return json.dumps(data, indent=2)

    @staticmethod
    def to_markdown(result: BenchmarkSuiteResult) -> str:
        """Convert results to Markdown table."""
        lines = [
            f"# Benchmark Results: {result.suite_name}",
            f"",
            f"**Date**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Time**: {result.total_time:.2f}s",
            f"",
            "| Benchmark | Iterations | Avg (ms) | Min (ms) | Max (ms) | Std Dev | Throughput (ops/s) |",
            "|-----------|------------|----------|----------|----------|---------|-------------------|"
        ]

        for r in result.results:
            lines.append(
                f"| {r.name} | {r.iterations} | {r.avg_time*1000:.2f} | "
                f"{r.min_time*1000:.2f} | {r.max_time*1000:.2f} | "
                f"{r.std_dev*1000:.2f} | {r.throughput:.2f} |"
            )

        return "\n".join(lines)

    @staticmethod
    def print_summary(result: BenchmarkSuiteResult) -> None:
        """Print a summary to console."""
        print(f"\n{'='*70}")
        print(f"Benchmark Suite: {result.suite_name}")
        print(f"{'='*70}")
        print(f"Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Benchmarks: {result.total_benchmarks}")
        print(f"Total Time: {result.total_time:.2f}s")
        print(f"{'='*70}")
        print(f"{'Name':<30} {'Avg (ms)':<12} {'Throughput':<15}")
        print(f"{'-'*70}")

        for r in result.results:
            print(f"{r.name:<30} {r.avg_time*1000:<12.2f} {r.throughput:<15.2f}")

        print(f"{'='*70}\n")


# Pre-defined benchmarks
class AgentBenchmarks:
    """Collection of agent benchmarks."""

    def __init__(self, agent: Any):
        self.agent = agent

    async def simple_task(self) -> None:
        """Benchmark simple task execution."""
        await self.agent.execute(task="What is 2+2?")

    async def code_generation(self) -> None:
        """Benchmark code generation."""
        await self.agent.execute(task="Write a Python hello world function")

    async def multi_step_task(self) -> None:
        """Benchmark multi-step task."""
        await self.agent.execute(
            task="List files, then create a file, then read it"
        )


class ToolBenchmarks:
    """Collection of tool benchmarks."""

    def __init__(self, tool_registry: Any):
        self.registry = tool_registry

    async def web_search(self) -> None:
        """Benchmark web search tool."""
        tool = self.registry.get("web_search")
        await tool.execute(query="Python programming")

    async def file_read(self, path: str) -> None:
        """Benchmark file reading."""
        tool = self.registry.get("file_reader")
        await tool.execute(path=path)


class GuardrailBenchmarks:
    """Collection of guardrail benchmarks."""

    def __init__(self, guardrail: Any):
        self.guardrail = guardrail

    async def input_validation(self) -> None:
        """Benchmark input validation."""
        await self.guardrail.validate("Test input message")

    async def pii_detection(self, text: str) -> None:
        """Benchmark PII detection."""
        await self.guardrail.detect_pii(text)


# Example benchmark suite creation
async def create_standard_benchmark_suite(agent: Any) -> BenchmarkSuiteResult:
    """Create and run a standard benchmark suite."""
    suite = BenchmarkSuite("Standard Benchmarks")

    # Add benchmarks
    async def simple_task():
        return {"result": "success"}

    async def medium_task():
        await asyncio.sleep(0.01)
        return {"result": "success"}

    async def complex_task():
        await asyncio.sleep(0.05)
        return {"result": "success"}

    suite.add("Simple Task", simple_task, iterations=100)
    suite.add("Medium Task", medium_task, iterations=50)
    suite.add("Complex Task", complex_task, iterations=20)

    return await suite.run_all()


if __name__ == "__main__":
    async def main():
        # Run example benchmarks
        result = await create_standard_benchmark_suite(None)
        BenchmarkReporter.print_summary(result)

        # Save reports
        json_report = BenchmarkReporter.to_json(result)
        md_report = BenchmarkReporter.to_markdown(result)

        print("\nJSON Report:")
        print(json_report)

        print("\nMarkdown Report:")
        print(md_report)

    asyncio.run(main())
