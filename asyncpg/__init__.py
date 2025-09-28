import asyncio


class Pool:  # pragma: no cover - simple compatibility stub
    async def close(self) -> None:  # pragma: no cover - stub
        return None


async def create_pool(*args, **kwargs) -> Pool:  # pragma: no cover - stub
    await asyncio.sleep(0)
    return Pool()
