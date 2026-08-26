"""RabbitMQ job queue — publish/consume with DLQ, priority, and bounded prefetch.

Backed by ``pika``. Mirrors ``KafkaConnector``'s circuit-breaker contract
(see ``dataenginex/data/connectors/kafka.py``): RabbitMQ being slow or down
must never block or crash the caller.

- ``publish`` catches connection/publish errors, logs, and swallows them
  (never raises).
- ``consume`` catches broker-down errors, logs, and returns 0 processed
  (never raises).
- ``health_check`` returns ``False`` (never raises) on any failure.

Example::

    queue = RabbitMQQueue(
        host="rabbitmq",
        queue_name="movie-enrichment",
        dlq_name="movie-enrichment.dlq",
        prefetch_count=10,
    )
    queue.publish({"movie_id": 12345}, priority=5)
    queue.consume(lambda msg: enrich(msg["movie_id"]))
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from typing import Any

import pika
import structlog

logger = structlog.get_logger()


class RabbitMQQueue:
    """Publish/consume job queue backed by RabbitMQ.

    Args:
        host: RabbitMQ broker hostname.
        port: RabbitMQ broker port.
        queue_name: Name of the work queue.
        dlq_name: Dead-letter queue name. When given, the work queue is
            declared with a DLX so nacked/expired messages route there
            instead of vanishing. When omitted, nacked messages are
            requeued instead (see ``consume`` docstring for the tradeoff).
        prefetch_count: Bounded concurrency — ``consume`` never pulls more
            than this many unacked messages in one batch.
        timeout_seconds: Connection/socket timeout for all broker calls.
    """

    def __init__(
        self,
        host: str,
        queue_name: str,
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/",
        dlq_name: str | None = None,
        prefetch_count: int = 10,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._virtual_host = virtual_host
        self._queue_name = queue_name
        self._dlq_name = dlq_name
        self._prefetch_count = prefetch_count
        self._timeout = timeout_seconds

    def _connect(self) -> pika.BlockingConnection:
        params = pika.ConnectionParameters(
            host=self._host,
            port=self._port,
            virtual_host=self._virtual_host,
            credentials=pika.PlainCredentials(self._username, self._password),
            blocked_connection_timeout=self._timeout,
            socket_timeout=self._timeout,
        )
        return pika.BlockingConnection(params)

    def _declare_topology(self, channel: Any) -> None:
        """Declare the work queue (and DLX/DLQ if configured). Idempotent."""
        queue_args: dict[str, Any] = {"x-max-priority": 9}
        if self._dlq_name is not None:
            dlx_name = f"{self._queue_name}.dlx"
            channel.exchange_declare(exchange=dlx_name, exchange_type="direct", durable=True)
            channel.queue_declare(queue=self._dlq_name, durable=True)
            channel.queue_bind(exchange=dlx_name, queue=self._dlq_name, routing_key=self._dlq_name)
            queue_args["x-dead-letter-exchange"] = dlx_name
            queue_args["x-dead-letter-routing-key"] = self._dlq_name
        channel.queue_declare(queue=self._queue_name, durable=True, arguments=queue_args)

    def publish(self, message: dict[str, Any], priority: int = 0) -> None:
        """JSON-serialize and publish with the given priority (0-9).

        Circuit breaker: broker down or any connection/publish error is
        caught, logged, and swallowed — never raised to the caller.
        """
        connection = None
        try:
            connection = self._connect()
            channel = connection.channel()
            self._declare_topology(channel)
            channel.basic_publish(
                exchange="",
                routing_key=self._queue_name,
                body=json.dumps(message).encode("utf-8"),
                properties=pika.BasicProperties(priority=priority, delivery_mode=2),
            )
            logger.debug("rabbitmq publish complete", queue=self._queue_name, priority=priority)
        except Exception as exc:  # noqa: BLE001 - broker down must never crash the caller
            logger.error("rabbitmq publish failed", queue=self._queue_name, error=str(exc))
        finally:
            if connection is not None:
                with contextlib.suppress(Exception):
                    connection.close()

    def _ack_or_nack(
        self,
        body: bytes,
        channel: pika.channel.Channel,
        method: pika.spec.Basic.Deliver,
        handler: Callable[[dict[str, Any]], bool],
    ) -> bool:
        try:
            ok = bool(handler(json.loads(body)))
        except Exception as exc:  # noqa: BLE001 - handler bugs must not crash the loop
            logger.error("rabbitmq handler raised", queue=self._queue_name, error=str(exc))
            ok = False
        if ok:
            channel.basic_ack(delivery_tag=method.delivery_tag)
        else:
            requeue = self._dlq_name is None
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=requeue)
        return ok

    def consume(self, handler: Callable[[dict[str, Any]], bool]) -> int:
        """Pull up to ``prefetch_count`` messages and hand each to ``handler``.

        For each message: ack if ``handler`` returns truthy (or doesn't
        raise); otherwise nack it. Nack routing:

        - DLQ configured: nack without requeue — RabbitMQ dead-letters the
          message to the DLQ per the queue's declared DLX.
        - No DLQ configured: nack with requeue=True (best-effort single
          redelivery). ponytail: no retry-count tracking, so a handler
          that always fails will loop the message forever without a DLQ —
          configure ``dlq_name`` for real workloads.

        Broker-down (or any error before/while pulling) is caught, logged,
        and returns the count of messages processed so far — never raises.

        Returns:
            Number of messages pulled and handed to ``handler`` this call.
        """
        connection = None
        processed = 0
        try:
            connection = self._connect()
            channel = connection.channel()
            channel.basic_qos(prefetch_count=self._prefetch_count)
            self._declare_topology(channel)
        except Exception as exc:  # noqa: BLE001 - broker down must never crash the caller
            logger.error(
                "rabbitmq consume: broker unreachable", queue=self._queue_name, error=str(exc)
            )
            if connection is not None:
                with contextlib.suppress(Exception):
                    connection.close()
            return 0

        try:
            for _ in range(self._prefetch_count):
                method, _properties, body = channel.basic_get(
                    queue=self._queue_name, auto_ack=False
                )
                if method is None:
                    break  # queue empty

                processed += 1
                self._ack_or_nack(body, channel, method, handler)
        except Exception as exc:  # noqa: BLE001 - broker down mid-batch must never crash the caller
            logger.error(
                "rabbitmq consume failed mid-batch", queue=self._queue_name, error=str(exc)
            )
        finally:
            with contextlib.suppress(Exception):
                connection.close()

        logger.debug("rabbitmq consume complete", queue=self._queue_name, processed=processed)
        return processed

    def health_check(self) -> bool:
        """Lightweight connection check. Returns False (never raises) on failure."""
        try:
            connection = self._connect()
            connection.close()
            return True
        except Exception:
            return False
