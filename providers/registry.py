from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar, Union

T = TypeVar("T")
Factory = Union[type, Callable[..., T]]


class Registry(Generic[T]):
    """Name → class/factory registry. Engines stay provider-agnostic."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Factory] = {}

    def register(self, *names: str) -> Callable[[Factory], Factory]:
        def deco(factory: Factory) -> Factory:
            self.add(factory, *names)
            return factory

        return deco

    def add(self, factory: Factory, *names: str) -> None:
        for name in names:
            self._factories[name.lower().replace("-", "_")] = factory

    def get(self, name: str, **kwargs: Any) -> T:
        key = name.lower().replace("-", "_")
        factory = self._factories.get(key)
        if factory is None:
            raise ValueError(
                f"Unknown {self.kind} adapter '{name}'. Known: {self.list()}"
            )
        return factory(**kwargs)  # type: ignore[misc]

    def list(self) -> list[str]:
        return sorted(set(self._factories))


realtime_registry: Registry = Registry("realtime")
stt_registry: Registry = Registry("stt")
tts_registry: Registry = Registry("tts")

AdapterRegistry = Registry
