"""Generic backend registry pattern.

Every subsystem (connectors, trackers, retrievers, etc.) uses a
``BackendRegistry`` to discover and instantiate backend implementations.

Usage::

    from dataenginex.runtime.registry import BackendRegistry

    connector_registry: BackendRegistry[BaseConnector] = BackendRegistry("connector")

    @connector_registry.decorator("csv")
    class CsvConnector(BaseConnector):
        ...

    # Later:
    cls = connector_registry.get("csv")
    instance = cls(**kwargs)
"""

from __future__ import annotations

from collections.abc import Callable

import structlog

logger = structlog.get_logger()


class BackendRegistry[T]:
    """Type-safe registry for backend implementations.

    Parameters:
        domain: Human-readable name for error messages (e.g. "connector").
    """

    def __init__(self, domain: str) -> None:
        self._domain = domain
        self._backends: dict[str, type[T]] = {}
        self._default: str | None = None

    def register(self, name: str, cls: type[T], *, is_default: bool = False) -> None:
        """Register a backend class under *name*.

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._backends:
            msg = f"{self._domain} backend '{name}' already registered"
            raise ValueError(msg)
        self._backends[name] = cls
        if is_default:
            self._default = name
        logger.debug(
            "backend registered",
            domain=self._domain,
            name=name,
            default=is_default,
        )

    def decorator(self, name: str, *, is_default: bool = False) -> Callable[[type[T]], type[T]]:
        """Class decorator for registration.

        Usage::

            @registry.decorator("csv")
            class CsvConnector(BaseConnector):
                ...
        """

        def wrapper(cls: type[T]) -> type[T]:
            self.register(name, cls, is_default=is_default)
            return cls

        return wrapper

    def get(self, name: str) -> type[T]:
        """Return the backend class registered under *name*.

        Raises:
            KeyError: If *name* is not registered.
        """
        try:
            return self._backends[name]
        except KeyError:
            available = ", ".join(sorted(self._backends)) or "(none)"
            msg = f"{self._domain} backend '{name}' not found. Available: {available}"
            raise KeyError(msg) from None

    def get_default(self) -> type[T]:
        """Return the default backend class.

        Raises:
            ValueError: If no default has been set.
        """
        if self._default is None:
            msg = f"{self._domain} registry has no default backend"
            raise ValueError(msg)
        return self._backends[self._default]

    def list(self) -> list[str]:
        """Return all registered backend names."""
        return list(self._backends.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._backends

    def __len__(self) -> int:
        return len(self._backends)
