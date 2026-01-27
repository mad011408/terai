"""
Parallel executor for concurrent task execution.
"""

from typing import Any, Dict, List, Optional, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid


@dataclass
class Task:
    """A task to be executed."""
    task_id: str
    name: str
    coroutine: Coroutine
    priority: int = 0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ParallelExecutor:
    """
    Executes tasks in parallel with concurrency control.
    """

    def __init__(self, max_concurrency: int = 10,
                 default_timeout: float = 60.0):
        self.max_concurrency = max_concurrency
        self.default_timeout = default_timeout
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.running_tasks: Dict[str, Task] = {}
        self.results: Dict[str, TaskResult] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrency)

    async def execute(self, tasks: List[Task],
                     fail_fast: bool = False) -> List[TaskResult]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of tasks to execute
            fail_fast: Stop on first failure if True

        Returns:
            List of task results
        """
        # Sort by priority (higher first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        if fail_fast:
            return await self._execute_fail_fast(sorted_tasks)
        else:
            return await self._execute_all(sorted_tasks)

    async def _execute_all(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute all tasks, collecting all results."""
        async def run_task(task: Task) -> TaskResult:
            async with self.semaphore:
                return await self._run_single_task(task)

        results = await asyncio.gather(
            *[run_task(task) for task in tasks],
            return_exceptions=True
        )

        return [
            r if isinstance(r, TaskResult) else TaskResult(
                task_id="unknown",
                success=False,
                error=str(r)
            )
            for r in results
        ]

    async def _execute_fail_fast(self, tasks: List[Task]) -> List[TaskResult]:
        """Execute tasks, stopping on first failure."""
        results = []
        pending = set()

        for task in tasks:
            # Create task
            coro = self._run_single_task(task)
            pending.add(asyncio.create_task(coro, name=task.task_id))

            # Limit concurrency
            while len(pending) >= self.max_concurrency:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED
                )

                for completed in done:
                    result = completed.result()
                    results.append(result)

                    if not result.success:
                        # Cancel remaining tasks
                        for p in pending:
                            p.cancel()
                        return results

        # Wait for remaining tasks
        if pending:
            done, _ = await asyncio.wait(pending)
            for completed in done:
                try:
                    result = completed.result()
                    results.append(result)
                except:
                    pass

        return results

    async def _run_single_task(self, task: Task) -> TaskResult:
        """Run a single task."""
        task_id = task.task_id
        self.running_tasks[task_id] = task

        started_at = datetime.now()
        timeout = task.timeout or self.default_timeout

        try:
            result = await asyncio.wait_for(task.coroutine, timeout=timeout)

            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=(datetime.now() - started_at).total_seconds(),
                started_at=started_at,
                completed_at=datetime.now()
            )

        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task_id,
                success=False,
                error=f"Task timed out after {timeout}s",
                execution_time=timeout,
                started_at=started_at,
                completed_at=datetime.now()
            )

        except Exception as e:
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - started_at).total_seconds(),
                started_at=started_at,
                completed_at=datetime.now()
            )

        finally:
            del self.running_tasks[task_id]

    async def map(self, func: Callable, items: List[Any],
                 timeout: Optional[float] = None) -> List[Any]:
        """
        Map a function over items in parallel.

        Args:
            func: Async function to apply
            items: Items to process
            timeout: Timeout per item

        Returns:
            List of results
        """
        tasks = [
            Task(
                task_id=str(uuid.uuid4()),
                name=f"map_{i}",
                coroutine=func(item) if asyncio.iscoroutinefunction(func) else asyncio.to_thread(func, item),
                timeout=timeout
            )
            for i, item in enumerate(items)
        ]

        results = await self.execute(tasks)
        return [r.result for r in results]

    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run a synchronous function in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            lambda: func(*args, **kwargs)
        )

    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """Get information about running tasks."""
        return [
            {
                "task_id": task.task_id,
                "name": task.name,
                "priority": task.priority
            }
            for task in self.running_tasks.values()
        ]

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.thread_pool.shutdown(wait=False)


class BatchExecutor:
    """
    Executes tasks in batches.
    """

    def __init__(self, batch_size: int = 10,
                 max_concurrency: int = 5):
        self.batch_size = batch_size
        self.executor = ParallelExecutor(max_concurrency=max_concurrency)

    async def execute_batches(self, tasks: List[Task],
                             on_batch_complete: Optional[Callable] = None) -> List[TaskResult]:
        """
        Execute tasks in batches.

        Args:
            tasks: All tasks to execute
            on_batch_complete: Callback after each batch

        Returns:
            All results
        """
        all_results = []

        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            batch_results = await self.executor.execute(batch)
            all_results.extend(batch_results)

            if on_batch_complete:
                await on_batch_complete(i // self.batch_size, batch_results)

        return all_results


class PriorityExecutor:
    """
    Executes tasks based on priority with preemption.
    """

    def __init__(self, max_concurrency: int = 5):
        self.max_concurrency = max_concurrency
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running = False
        self.workers: List[asyncio.Task] = []

    async def submit(self, task: Task) -> None:
        """Submit a task to the queue."""
        # Priority queue uses min-heap, so negate priority for max-priority first
        await self.task_queue.put((-task.priority, task))

    async def start(self) -> None:
        """Start worker tasks."""
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_concurrency)
        ]

    async def stop(self) -> None:
        """Stop all workers."""
        self.running = False
        for worker in self.workers:
            worker.cancel()

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine."""
        while self.running:
            try:
                priority, task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )

                try:
                    await task.coroutine
                except Exception as e:
                    pass  # Handle error

                self.task_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
