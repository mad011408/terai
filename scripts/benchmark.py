#!/usr/bin/env python3
"""
AI Terminal Agent - Benchmark Script
Runs performance benchmarks and generates reports.
"""

import asyncio
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.evals.benchmarks import (
    Benchmark,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    BenchmarkReporter
)


class AgentBenchmark:
    """Benchmarks for agent operations."""

    def __init__(self):
        self.results = []

    async def benchmark_simple_task(self) -> float:
        """Benchmark a simple task."""
        start = time.perf_counter()
        # Simulate simple task
        await asyncio.sleep(0.01)
        return time.perf_counter() - start

    async def benchmark_code_generation(self) -> float:
        """Benchmark code generation."""
        start = time.perf_counter()
        # Simulate code generation
        await asyncio.sleep(0.05)
        return time.perf_counter() - start

    async def benchmark_research_task(self) -> float:
        """Benchmark research task."""
        start = time.perf_counter()
        # Simulate research
        await asyncio.sleep(0.1)
        return time.perf_counter() - start


class ToolBenchmark:
    """Benchmarks for tool operations."""

    async def benchmark_web_search(self) -> float:
        """Benchmark web search."""
        start = time.perf_counter()
        # Simulate web search
        await asyncio.sleep(0.02)
        return time.perf_counter() - start

    async def benchmark_file_read(self) -> float:
        """Benchmark file reading."""
        start = time.perf_counter()
        # Simulate file read
        await asyncio.sleep(0.005)
        return time.perf_counter() - start

    async def benchmark_terminal_execution(self) -> float:
        """Benchmark terminal execution."""
        start = time.perf_counter()
        # Simulate terminal command
        await asyncio.sleep(0.03)
        return time.perf_counter() - start


class MemoryBenchmark:
    """Benchmarks for memory operations."""

    async def benchmark_cache_read(self) -> float:
        """Benchmark cache read."""
        start = time.perf_counter()
        # Simulate cache read
        await asyncio.sleep(0.001)
        return time.perf_counter() - start

    async def benchmark_cache_write(self) -> float:
        """Benchmark cache write."""
        start = time.perf_counter()
        # Simulate cache write
        await asyncio.sleep(0.002)
        return time.perf_counter() - start

    async def benchmark_vector_search(self) -> float:
        """Benchmark vector search."""
        start = time.perf_counter()
        # Simulate vector search
        await asyncio.sleep(0.05)
        return time.perf_counter() - start


class GuardrailBenchmark:
    """Benchmarks for guardrail operations."""

    async def benchmark_input_validation(self) -> float:
        """Benchmark input validation."""
        start = time.perf_counter()
        # Simulate validation
        await asyncio.sleep(0.003)
        return time.perf_counter() - start

    async def benchmark_pii_detection(self) -> float:
        """Benchmark PII detection."""
        start = time.perf_counter()
        # Simulate PII detection
        await asyncio.sleep(0.01)
        return time.perf_counter() - start

    async def benchmark_safety_classification(self) -> float:
        """Benchmark safety classification."""
        start = time.perf_counter()
        # Simulate classification
        await asyncio.sleep(0.02)
        return time.perf_counter() - start


async def run_agent_benchmarks(iterations: int = 100) -> BenchmarkSuiteResult:
    """Run agent benchmarks."""
    suite = BenchmarkSuite("Agent Benchmarks")
    bench = AgentBenchmark()

    suite.add("Simple Task", bench.benchmark_simple_task, iterations)
    suite.add("Code Generation", bench.benchmark_code_generation, iterations)
    suite.add("Research Task", bench.benchmark_research_task, iterations // 2)

    return await suite.run_all()


async def run_tool_benchmarks(iterations: int = 100) -> BenchmarkSuiteResult:
    """Run tool benchmarks."""
    suite = BenchmarkSuite("Tool Benchmarks")
    bench = ToolBenchmark()

    suite.add("Web Search", bench.benchmark_web_search, iterations)
    suite.add("File Read", bench.benchmark_file_read, iterations * 2)
    suite.add("Terminal Execution", bench.benchmark_terminal_execution, iterations)

    return await suite.run_all()


async def run_memory_benchmarks(iterations: int = 100) -> BenchmarkSuiteResult:
    """Run memory benchmarks."""
    suite = BenchmarkSuite("Memory Benchmarks")
    bench = MemoryBenchmark()

    suite.add("Cache Read", bench.benchmark_cache_read, iterations * 5)
    suite.add("Cache Write", bench.benchmark_cache_write, iterations * 2)
    suite.add("Vector Search", bench.benchmark_vector_search, iterations)

    return await suite.run_all()


async def run_guardrail_benchmarks(iterations: int = 100) -> BenchmarkSuiteResult:
    """Run guardrail benchmarks."""
    suite = BenchmarkSuite("Guardrail Benchmarks")
    bench = GuardrailBenchmark()

    suite.add("Input Validation", bench.benchmark_input_validation, iterations * 2)
    suite.add("PII Detection", bench.benchmark_pii_detection, iterations)
    suite.add("Safety Classification", bench.benchmark_safety_classification, iterations)

    return await suite.run_all()


async def run_all_benchmarks(iterations: int = 100) -> List[BenchmarkSuiteResult]:
    """Run all benchmark suites."""
    results = []

    print("Running Agent Benchmarks...")
    results.append(await run_agent_benchmarks(iterations))

    print("Running Tool Benchmarks...")
    results.append(await run_tool_benchmarks(iterations))

    print("Running Memory Benchmarks...")
    results.append(await run_memory_benchmarks(iterations))

    print("Running Guardrail Benchmarks...")
    results.append(await run_guardrail_benchmarks(iterations))

    return results


def generate_report(results: List[BenchmarkSuiteResult], output_dir: str) -> None:
    """Generate benchmark reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate individual reports
    for result in results:
        # Print summary
        BenchmarkReporter.print_summary(result)

        # JSON report
        json_path = output_path / f"{result.suite_name.lower().replace(' ', '_')}_{timestamp}.json"
        json_report = BenchmarkReporter.to_json(result)
        with open(json_path, 'w') as f:
            f.write(json_report)

        # Markdown report
        md_path = output_path / f"{result.suite_name.lower().replace(' ', '_')}_{timestamp}.md"
        md_report = BenchmarkReporter.to_markdown(result)
        with open(md_path, 'w') as f:
            f.write(md_report)

    # Combined summary
    combined = {
        "timestamp": datetime.now().isoformat(),
        "suites": []
    }

    for result in results:
        suite_data = {
            "name": result.suite_name,
            "total_benchmarks": result.total_benchmarks,
            "total_time": result.total_time,
            "benchmarks": [
                {
                    "name": r.name,
                    "avg_ms": r.avg_time * 1000,
                    "throughput": r.throughput
                }
                for r in result.results
            ]
        }
        combined["suites"].append(suite_data)

    summary_path = output_path / f"benchmark_summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\nReports saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AI Terminal Agent benchmarks"
    )

    parser.add_argument(
        "-s", "--suite",
        choices=["all", "agent", "tool", "memory", "guardrail"],
        default="all",
        help="Benchmark suite to run (default: all)"
    )

    parser.add_argument(
        "-i", "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)"
    )

    parser.add_argument(
        "-o", "--output",
        default="benchmark_reports",
        help="Output directory for reports (default: benchmark_reports)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (fewer iterations)"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    iterations = args.iterations
    if args.quick:
        iterations = 10

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║            AI Terminal Agent - Benchmark Suite                ║
╚══════════════════════════════════════════════════════════════╝

Configuration:
  Suite: {args.suite}
  Iterations: {iterations}
  Output: {args.output}
""")

    results = []

    if args.suite == "all":
        results = await run_all_benchmarks(iterations)
    elif args.suite == "agent":
        results = [await run_agent_benchmarks(iterations)]
    elif args.suite == "tool":
        results = [await run_tool_benchmarks(iterations)]
    elif args.suite == "memory":
        results = [await run_memory_benchmarks(iterations)]
    elif args.suite == "guardrail":
        results = [await run_guardrail_benchmarks(iterations)]

    if args.json:
        # Output JSON to stdout
        combined = []
        for result in results:
            combined.append(json.loads(BenchmarkReporter.to_json(result)))
        print(json.dumps(combined, indent=2))
    else:
        # Generate reports
        generate_report(results, args.output)

        # Print overall summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)

        total_benchmarks = sum(r.total_benchmarks for r in results)
        total_time = sum(r.total_time for r in results)

        print(f"Total Suites: {len(results)}")
        print(f"Total Benchmarks: {total_benchmarks}")
        print(f"Total Time: {total_time:.2f}s")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
