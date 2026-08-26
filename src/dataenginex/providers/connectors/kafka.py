"""Kafka connector — produce/consume with a non-blocking circuit breaker.

Registered as ``type: kafka`` in dex.yaml sources.  Backed by
``confluent-kafka`` (librdkafka).  Kafka being slow or down must never
block or crash the pipeline request path:

- Produce is fire-and-forget into a small bounded local buffer; broker
  errors are caught, logged, and swallowed (never raised).
- Consume returns ``[]`` on broker errors instead of raising.
- ``health_check`` returns ``False`` (never raises) on any failure.

Example dex.yaml::

    sources:
      movie_changes_producer:
        type: kafka
        connection:
          bootstrap_servers: "kafka:9092"
          topic: movie-changes
          mode: produce

      movie_changes_consumer:
        type: kafka
        connection:
          bootstrap_servers: "kafka:9092"
          topic: movie-changes
          mode: consume
          group_id: movie-changes-consumer
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import structlog
from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient

from dataenginex.foundation.plugin_contracts import BaseConnector
from dataenginex.providers.connectors import connector_registry

logger = structlog.get_logger()


@connector_registry.decorator("kafka")
class KafkaConnector(BaseConnector):
    """Produce/consume connector backed by confluent-kafka.

    Args:
        bootstrap_servers: Kafka broker list (e.g. ``"kafka:9092"``).
        topic: Topic to produce to or consume from.
        mode: ``"produce"`` or ``"consume"``.
        group_id: Consumer group ID — required when ``mode="consume"``.
        timeout_seconds: Timeout for broker calls (poll, metadata, flush).
        max_buffer: Cap on the local fire-and-forget produce buffer. When
            full, the oldest pending record is dropped (never blocks).
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        mode: str,
        group_id: str | None = None,
        timeout_seconds: float = 10.0,
        max_buffer: int = 1000,
        spool_path: str | None = None,
        dlq_path: str | None = None,
        security_protocol: str = "PLAINTEXT",
        sasl_mechanism: str = "PLAIN",
        username: str = "",
        password: str = "",
        **kwargs: Any,
    ) -> None:
        if mode not in ("produce", "consume"):
            msg = f"KafkaConnector mode must be 'produce' or 'consume', got {mode!r}"
            raise ValueError(msg)
        if mode == "consume" and not group_id:
            msg = "KafkaConnector requires group_id when mode='consume'"
            raise ValueError(msg)

        self._bootstrap_servers = bootstrap_servers
        self._topic = topic
        self._mode = mode
        self._group_id = group_id
        self._timeout = timeout_seconds
        self._max_buffer = max_buffer
        self._spool_path = Path(spool_path) if spool_path else None
        self._dlq_path = Path(dlq_path) if dlq_path else None
        self._security_protocol = security_protocol
        self._sasl_mechanism = sasl_mechanism
        self._username = username
        self._password = password
        self._buffer: deque[dict[str, Any]] = deque()
        self._inflight: dict[int, dict[str, Any]] = {}
        self._next_delivery_id = 0
        self._producer: Producer | None = None
        self._consumer: Consumer | None = None

    def connect(self) -> None:
        self._load_spool()
        client_config: dict[str, Any] = {
            "bootstrap.servers": self._bootstrap_servers,
            "security.protocol": self._security_protocol,
        }
        if self._security_protocol.startswith("SASL"):
            client_config.update(
                {
                    "sasl.mechanism": self._sasl_mechanism,
                    "sasl.username": self._username,
                    "sasl.password": self._password,
                }
            )
        if self._mode == "produce":
            self._producer = Producer(client_config)
        else:
            self._consumer = Consumer(
                {
                    **client_config,
                    "group.id": self._group_id,
                    # "earliest": this connector backs scheduled, independently
                    # run producer/consumer pipelines (not a long-lived consumer
                    # group) — "latest" would skip everything published before
                    # this exact run subscribes. Auto-commit (confluent-kafka's
                    # default) advances the offset per run, so later runs still
                    # only see what's new since the last one.
                    "auto.offset.reset": "earliest",
                }
            )
            self._consumer.subscribe([self._topic])
        logger.debug("kafka connector ready", topic=self._topic, mode=self._mode)

    def _load_spool(self) -> None:
        if self._spool_path is None or not self._spool_path.exists():
            return
        try:
            for line in self._spool_path.read_text().splitlines():
                if line.strip():
                    self._buffer.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            logger.warning("kafka spool could not be loaded", error=str(exc))

    def _persist_spool(self) -> None:
        if self._spool_path is None:
            return
        self._spool_path.parent.mkdir(parents=True, exist_ok=True)
        pending = [*self._inflight.values(), *self._buffer]
        payload = "".join(f"{json.dumps(record)}\n" for record in pending)
        temp_path = self._spool_path.with_suffix(self._spool_path.suffix + ".tmp")
        temp_path.write_text(payload)
        temp_path.replace(self._spool_path)

    def _dead_letter(self, record: dict[str, Any]) -> None:
        if self._dlq_path is None:
            return
        self._dlq_path.parent.mkdir(parents=True, exist_ok=True)
        with self._dlq_path.open("a") as handle:
            handle.write(f"{json.dumps(record)}\n")

    def _delivery_callback(self, delivery_id: int, error: Any, _message: Any) -> None:
        record = self._inflight.pop(delivery_id, None)
        if record is None:
            return
        if error is not None:
            self._buffer.appendleft(record)
            logger.error(
                "kafka delivery failed, record returned to local buffer",
                topic=self._topic,
                error=str(error),
            )
        self._persist_spool()

    def _drain_buffer(self) -> None:
        if self._producer is None or not self._buffer:
            return
        pending = list(self._buffer)
        self._buffer.clear()
        for index, record in enumerate(pending):
            if len(self._inflight) >= self._max_buffer:
                self._buffer.extend(pending[index:])
                break
            delivery_id = self._next_delivery_id
            self._next_delivery_id += 1
            self._inflight[delivery_id] = record

            def on_delivery(
                error: Any,
                message: Any,
                key: int = delivery_id,
            ) -> None:
                self._delivery_callback(key, error, message)

            try:
                self._producer.produce(
                    self._topic,
                    value=json.dumps(record).encode("utf-8"),
                    on_delivery=on_delivery,
                )
                self._producer.poll(0)
            except Exception as exc:  # noqa: BLE001
                self._inflight.pop(delivery_id, None)
                self._buffer.append(record)
                self._buffer.extend(pending[index + 1 :])
                logger.error(
                    "kafka connector produce failed, record kept in local buffer",
                    topic=self._topic,
                    error=str(exc),
                )
                break

    def disconnect(self) -> None:
        if self._producer is not None:
            try:
                self._producer.flush(self._timeout)
            except Exception as exc:  # noqa: BLE001 - never crash on shutdown
                logger.warning("kafka producer flush failed", error=str(exc))
            self._producer = None
        if self._consumer is not None:
            try:
                self._consumer.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("kafka consumer close failed", error=str(exc))
            self._consumer = None
        self._persist_spool()

    def write(self, data: Any, *, table: str = "", **kwargs: Any) -> None:
        """Fire-and-forget produce. Never blocks or raises on broker errors."""
        if self._mode != "produce":
            raise NotImplementedError("KafkaConnector is consume-only in mode='consume'")
        if self._producer is None:
            msg = "KafkaConnector not connected — call connect() first"
            raise RuntimeError(msg)

        records = data if isinstance(data, list) else [data]

        for record in records:
            if len(self._buffer) + len(self._inflight) >= self._max_buffer:
                dropped = self._buffer.popleft() if self._buffer else record
                self._dead_letter(dropped)
                logger.warning(
                    "kafka connector local buffer full, dropping oldest record",
                    topic=self._topic,
                    dropped=dropped,
                )
                if dropped is record:
                    continue
            self._buffer.append(record)
            self._drain_buffer()
            self._persist_spool()

    def read(self, *, table: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        """Poll available messages for up to timeout_seconds. Never raises on broker errors."""
        if self._mode != "consume":
            raise NotImplementedError("KafkaConnector is produce-only in mode='produce'")
        if self._consumer is None:
            msg = "KafkaConnector not connected — call connect() first"
            raise RuntimeError(msg)

        records: list[dict[str, Any]] = []
        try:
            msgs = self._consumer.consume(num_messages=500, timeout=self._timeout)
            for kafka_msg in msgs:
                if kafka_msg.error():
                    logger.error("kafka connector consume error", error=str(kafka_msg.error()))
                    continue
                try:
                    val = kafka_msg.value()
                    if val is None:
                        continue  # tombstone
                    records.append(json.loads(val.decode("utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning("kafka connector dropped unparseable message", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - broker down must return [] not raise
            logger.error("kafka connector broker unreachable", topic=self._topic, error=str(exc))
            return []

        logger.debug("kafka connector read complete", topic=self._topic, records=len(records))
        return records

    def health_check(self) -> bool:
        try:
            config: dict[str, Any] = {
                "bootstrap.servers": self._bootstrap_servers,
                "security.protocol": self._security_protocol,
            }
            if self._security_protocol.startswith("SASL"):
                config.update(
                    {
                        "sasl.mechanism": self._sasl_mechanism,
                        "sasl.username": self._username,
                        "sasl.password": self._password,
                    }
                )
            client = AdminClient(config)
            metadata = client.list_topics(timeout=self._timeout)
            return metadata is not None
        except Exception:
            return False
