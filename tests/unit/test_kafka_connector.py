"""KafkaConnector tests — mocked confluent_kafka client, no real broker required."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dataenginex.providers.connectors.kafka import KafkaConnector


def _msg(payload: dict, error=None) -> MagicMock:
    m = MagicMock()
    m.error.return_value = error
    m.value.return_value = json.dumps(payload).encode("utf-8")
    return m


class TestKafkaConnectorConstruction:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            KafkaConnector(bootstrap_servers="localhost:9092", topic="t", mode="bogus")

    def test_consume_without_group_id_raises(self) -> None:
        with pytest.raises(ValueError, match="group_id"):
            KafkaConnector(bootstrap_servers="localhost:9092", topic="t", mode="consume")


class TestKafkaConnectorProduce:
    @patch("dataenginex.providers.connectors.kafka.Producer")
    def test_successful_produce(self, mock_producer_cls) -> None:
        mock_producer = MagicMock()
        mock_producer_cls.return_value = mock_producer

        conn = KafkaConnector(bootstrap_servers="localhost:9092", topic="t", mode="produce")
        conn.connect()
        conn.write([{"id": 1}, {"id": 2}])

        assert mock_producer.produce.call_count == 2
        assert len(conn._buffer) == 0  # delivered records are popped

    @patch("dataenginex.providers.connectors.kafka.Producer")
    def test_produce_during_broker_down_does_not_raise(self, mock_producer_cls) -> None:
        mock_producer = MagicMock()
        mock_producer.produce.side_effect = Exception("broker down")
        mock_producer_cls.return_value = mock_producer

        conn = KafkaConnector(bootstrap_servers="localhost:9092", topic="t", mode="produce")
        conn.connect()

        # Must not raise even though every produce() call fails.
        conn.write([{"id": 1}])

        # Failed record stays buffered rather than being silently lost.
        assert len(conn._buffer) == 1

    @patch("dataenginex.providers.connectors.kafka.Producer")
    def test_async_delivery_failure_returns_record_to_durable_buffer(
        self, mock_producer_cls: MagicMock, tmp_path: Path
    ) -> None:
        producer = MagicMock()
        callbacks = []

        def capture(_topic, *, value, on_delivery):
            callbacks.append(on_delivery)

        producer.produce.side_effect = capture
        mock_producer_cls.return_value = producer
        spool = tmp_path / "kafka.jsonl"
        conn = KafkaConnector(
            bootstrap_servers="localhost:9092",
            topic="t",
            mode="produce",
            spool_path=str(spool),
        )
        conn.connect()
        conn.write({"id": 8})

        assert len(conn._inflight) == 1
        callbacks[0]("broker rejected record", MagicMock())

        assert list(conn._buffer) == [{"id": 8}]
        assert conn._inflight == {}
        assert json.loads(spool.read_text()) == {"id": 8}

    @patch("dataenginex.providers.connectors.kafka.Producer")
    def test_buffer_overflow_drops_oldest_without_blocking(self, mock_producer_cls) -> None:
        mock_producer = MagicMock()
        # Every produce() fails so records accumulate in the local buffer.
        mock_producer.produce.side_effect = Exception("broker down")
        mock_producer_cls.return_value = mock_producer

        conn = KafkaConnector(
            bootstrap_servers="localhost:9092", topic="t", mode="produce", max_buffer=3
        )
        conn.connect()

        conn.write([{"id": i} for i in range(5)])

        assert len(conn._buffer) == 3
        assert [r["id"] for r in conn._buffer] == [2, 3, 4]

    @patch("dataenginex.providers.connectors.kafka.Producer")
    def test_write_not_connected_raises(self, mock_producer_cls) -> None:
        conn = KafkaConnector(bootstrap_servers="localhost:9092", topic="t", mode="produce")
        with pytest.raises(RuntimeError, match="not connected"):
            conn.write([{"id": 1}])

    @patch("dataenginex.providers.connectors.kafka.Producer")
    def test_write_in_consume_mode_raises(self, mock_producer_cls) -> None:
        conn = KafkaConnector(
            bootstrap_servers="localhost:9092", topic="t", mode="consume", group_id="g"
        )
        with pytest.raises(NotImplementedError):
            conn.write([{"id": 1}])

    @patch("dataenginex.providers.connectors.kafka.Producer")
    def test_failed_records_survive_restart_in_spool(
        self, mock_producer_cls: MagicMock, tmp_path: Path
    ) -> None:
        producer = MagicMock()
        producer.produce.side_effect = Exception("broker down")
        mock_producer_cls.return_value = producer
        spool = tmp_path / "kafka.jsonl"

        first = KafkaConnector(
            bootstrap_servers="localhost:9092",
            topic="t",
            mode="produce",
            spool_path=str(spool),
        )
        first.connect()
        first.write({"id": 7})
        first.disconnect()

        second = KafkaConnector(
            bootstrap_servers="localhost:9092",
            topic="t",
            mode="produce",
            spool_path=str(spool),
        )
        second.connect()
        assert list(second._buffer) == [{"id": 7}]

    @patch("dataenginex.providers.connectors.kafka.Producer")
    def test_sasl_credentials_are_forwarded(self, mock_producer_cls: MagicMock) -> None:
        connector = KafkaConnector(
            bootstrap_servers="kafka:9092",
            topic="t",
            mode="produce",
            security_protocol="SASL_PLAINTEXT",
            username="dex",
            password="secret",
        )
        connector.connect()

        config = mock_producer_cls.call_args.args[0]
        assert config["security.protocol"] == "SASL_PLAINTEXT"
        assert config["sasl.username"] == "dex"


class TestKafkaConnectorConsume:
    @patch("dataenginex.providers.connectors.kafka.Consumer")
    def test_successful_consume_returns_parsed_records(self, mock_consumer_cls) -> None:
        mock_consumer = MagicMock()
        mock_consumer.consume.return_value = [
            _msg({"id": 1}),
            _msg({"id": 2}),
        ]
        mock_consumer_cls.return_value = mock_consumer

        conn = KafkaConnector(
            bootstrap_servers="localhost:9092", topic="t", mode="consume", group_id="g"
        )
        conn.connect()
        result = conn.read()

        assert result == [{"id": 1}, {"id": 2}]
        mock_consumer.subscribe.assert_called_once_with(["t"])

    @patch("dataenginex.providers.connectors.kafka.Consumer")
    def test_consume_during_broker_down_returns_empty_list(self, mock_consumer_cls) -> None:
        mock_consumer = MagicMock()
        mock_consumer.consume.side_effect = Exception("broker down")
        mock_consumer_cls.return_value = mock_consumer

        conn = KafkaConnector(
            bootstrap_servers="localhost:9092", topic="t", mode="consume", group_id="g"
        )
        conn.connect()

        assert conn.read() == []

    @patch("dataenginex.providers.connectors.kafka.Consumer")
    def test_consume_skips_message_level_errors(self, mock_consumer_cls) -> None:
        mock_consumer = MagicMock()
        mock_consumer.consume.return_value = [_msg({}, error="partition error"), _msg({"id": 5})]
        mock_consumer_cls.return_value = mock_consumer

        conn = KafkaConnector(
            bootstrap_servers="localhost:9092", topic="t", mode="consume", group_id="g"
        )
        conn.connect()

        assert conn.read() == [{"id": 5}]

    @patch("dataenginex.providers.connectors.kafka.Consumer")
    def test_read_in_produce_mode_raises(self, mock_consumer_cls) -> None:
        conn = KafkaConnector(bootstrap_servers="localhost:9092", topic="t", mode="produce")
        with pytest.raises(NotImplementedError):
            conn.read()


class TestKafkaConnectorHealthCheck:
    @patch("dataenginex.providers.connectors.kafka.AdminClient")
    def test_health_check_true_on_success(self, mock_admin_cls) -> None:
        mock_admin = MagicMock()
        mock_admin.list_topics.return_value = MagicMock()
        mock_admin_cls.return_value = mock_admin

        conn = KafkaConnector(bootstrap_servers="localhost:9092", topic="t", mode="produce")
        assert conn.health_check() is True

    @patch("dataenginex.providers.connectors.kafka.AdminClient")
    def test_health_check_false_on_failure(self, mock_admin_cls) -> None:
        mock_admin = MagicMock()
        mock_admin.list_topics.side_effect = Exception("unreachable")
        mock_admin_cls.return_value = mock_admin

        conn = KafkaConnector(bootstrap_servers="localhost:9092", topic="t", mode="produce")
        assert conn.health_check() is False
