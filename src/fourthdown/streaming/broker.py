"""The broker behind the topics: Redpanda/Kafka, or an in-process stand-in.

Both give the producer and consumer the same small interface, so the replay, the scorer,
and the feed never know which one they run on. The Kafka settings are the ones the
delivery guarantees rest on:

* producer: ``enable.idempotence`` (no duplicates from producer retries) and ``acks=all``;
* consumer: ``enable.auto.commit`` off, so an offset is committed only after its result
  was delivered downstream (at least once), and ``auto.offset.reset=earliest``, so a new
  consumer group starts from the beginning of the replay.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Message:
    topic: str
    key: str
    value: bytes
    offset: int
    partition: int = 0


class Producer(Protocol):
    def send(self, topic: str, key: str, value: bytes) -> None: ...

    def flush(self) -> None: ...


class Consumer(Protocol):
    def poll(self, timeout: float) -> Message | None: ...

    def commit(self, message: Message) -> None: ...

    def close(self) -> None: ...


class Broker(Protocol):
    def producer(self) -> Producer: ...

    def consumer(self, topics: Iterable[str], group: str) -> Consumer: ...


# --- in-process --------------------------------------------------------------------------


class MemoryBroker:
    """One partition per topic, held in lists; consumer groups commit offsets like Kafka's.

    What the tests and ``fourthdown stream demo`` run on. A consumer that is closed
    without committing a message gets it again, as from Kafka, which is how the tests
    show the scorer's at-least-once delivery.
    """

    def __init__(self) -> None:
        self.topics: dict[str, list[Message]] = defaultdict(list)
        self.committed: dict[tuple[str, str], int] = {}

    def producer(self) -> MemoryProducer:
        return MemoryProducer(self)

    def consumer(self, topics: Iterable[str], group: str) -> MemoryConsumer:
        return MemoryConsumer(self, list(topics), group)

    def values(self, topic: str) -> list[bytes]:
        return [message.value for message in self.topics[topic]]


class MemoryProducer:
    def __init__(self, broker: MemoryBroker) -> None:
        self._broker = broker

    def send(self, topic: str, key: str, value: bytes) -> None:
        log = self._broker.topics[topic]
        log.append(Message(topic, key, value, offset=len(log)))

    def flush(self) -> None:
        return None


class MemoryConsumer:
    def __init__(self, broker: MemoryBroker, topics: list[str], group: str) -> None:
        self._broker = broker
        self._group = group
        # Resume after the group's last committed offset, as a Kafka consumer group does.
        self._positions = {topic: broker.committed.get((group, topic), -1) + 1 for topic in topics}

    def poll(self, timeout: float) -> Message | None:
        for topic, position in self._positions.items():
            log = self._broker.topics[topic]
            if position < len(log):
                self._positions[topic] = position + 1
                return log[position]
        return None

    def commit(self, message: Message) -> None:
        self._broker.committed[(self._group, message.topic)] = message.offset

    def close(self) -> None:
        return None


# --- Redpanda / Kafka --------------------------------------------------------------------


def producer_config(bootstrap: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": bootstrap,
        "enable.idempotence": True,
        "acks": "all",
        "linger.ms": 5,
        "client.id": "fourthdown",
    }


def consumer_config(bootstrap: str, group: str) -> dict[str, Any]:
    return {
        "bootstrap.servers": bootstrap,
        "group.id": group,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "client.id": "fourthdown",
    }


class StreamError(RuntimeError):
    """The broker reported an error: unreachable, a failed delivery, a consumer error."""


class KafkaBroker:
    """Redpanda or any Kafka cluster at ``bootstrap`` (``host:port``)."""

    def __init__(self, bootstrap: str) -> None:
        self.bootstrap = bootstrap

    def producer(self) -> KafkaProducer:
        return KafkaProducer(self.bootstrap)

    def consumer(self, topics: Iterable[str], group: str) -> KafkaConsumer:
        return KafkaConsumer(self.bootstrap, list(topics), group)


class KafkaProducer:
    def __init__(self, bootstrap: str) -> None:
        from confluent_kafka import Producer as ConfluentProducer

        self._producer = ConfluentProducer(producer_config(bootstrap))
        self._errors: list[str] = []

    def _delivered(self, error: Any, message: Any) -> None:
        if error is not None:
            self._errors.append(str(error))

    def send(self, topic: str, key: str, value: bytes) -> None:
        self._producer.produce(topic, key=key.encode(), value=value, on_delivery=self._delivered)
        self._producer.poll(0)

    def flush(self) -> None:
        remaining = self._producer.flush(30)
        if remaining or self._errors:
            errors, self._errors = self._errors, []
            raise StreamError(
                f"{remaining} message(s) undelivered; " + "; ".join(errors[:3])
                if errors
                else f"{remaining} message(s) undelivered after 30s"
            )


class KafkaConsumer:
    def __init__(self, bootstrap: str, topics: list[str], group: str) -> None:
        from confluent_kafka import Consumer as ConfluentConsumer

        self._consumer = ConfluentConsumer(consumer_config(bootstrap, group))
        self._consumer.subscribe(topics)
        self._raw: dict[tuple[str, int, int], Any] = {}

    def poll(self, timeout: float) -> Message | None:
        raw = self._consumer.poll(timeout)
        if raw is None:
            return None
        error = raw.error()
        if error is not None:
            from confluent_kafka import KafkaError

            if error.code() == KafkaError._PARTITION_EOF:
                return None
            raise StreamError(str(error))
        key = raw.key()
        message = Message(
            topic=str(raw.topic()),
            key=key.decode() if isinstance(key, bytes) else str(key or ""),
            value=bytes(raw.value() or b""),
            offset=int(raw.offset() or 0),
            partition=int(raw.partition() or 0),
        )
        self._raw[(message.topic, message.partition, message.offset)] = raw
        return message

    def commit(self, message: Message) -> None:
        raw = self._raw.pop((message.topic, message.partition, message.offset))
        self._consumer.commit(message=raw, asynchronous=False)

    def close(self) -> None:
        self._consumer.close()
