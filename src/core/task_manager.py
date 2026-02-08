import asyncio
import time
import inspect
from typing import Any, Callable
from src.utils.logger import Logger

logger = Logger.get_logger()


class TaskManager:
    """
    Handles queued async execution of heavy chatbot tasks.
    """

    def __init__(self):
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.worker_running: bool = False
        self.worker_task: asyncio.Task | None = None

    async def start(self):
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._task_worker())

    async def stop(self):
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                logger.info("Task worker cancelled")

    async def enqueue(
        self,
        coro_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Enqueue an async callable and await its result.
        """
        if not asyncio.iscoroutinefunction(coro_func):
            raise TypeError(f"{coro_func.__name__} must be async")

        sig = inspect.signature(coro_func)
        sig.bind(*args, **kwargs)  # validation only

        future = asyncio.get_running_loop().create_future()
        await self.task_queue.put((coro_func, args, kwargs, future))

        if not self.worker_running:
            await self.start()

        return await future

    async def _task_worker(self):
        if self.worker_running:
            return

        self.worker_running = True
        logger.info("Task worker started")

        try:
            while not self.task_queue.empty():
                queue_size = self.task_queue.qsize()
                start = time.time()

                coro_func, args, kwargs, future = await self.task_queue.get()

                try:
                    result = await coro_func(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    logger.error("Task error: %s", e)
                    future.set_exception(e)

                elapsed = time.time() - start
                logger.info(
                    "Task done in %.2fs | Queue before: %d | after: %d",
                    elapsed,
                    queue_size,
                    self.task_queue.qsize(),
                )

                self.task_queue.task_done()
        finally:
            self.worker_running = False
            logger.info("Task worker idle")
