from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from django.core.management import CommandParser
from django.core.management.base import BaseCommand
from django.db import models, transaction
from django.db.models import (
    Count,
    F,
    IntegerField,
    OuterRef,
    Q,
    QuerySet,
    Subquery,
    Sum,
)
from django.db.models.fields.json import KT
from django.db.models.functions import Cast, TruncDate
from django.utils import timezone

from lunes_cms.analytics.models import (
    AnalyticsEvent,
    DropoutAggregate,
    JobSelectionAggregate,
    ModuleDurationAggregate,
    SessionAggregate,
)

logger = logging.getLogger(__name__)


class EventAggregator(ABC):
    """
    Base class for event type aggregators.
    Subclasses implement aggregate() to transform raw events into aggregate models
    and to mark the consumed events with ``aggregated_at`` so they are not picked
    up on subsequent runs.
    """

    event_types: list[str]

    @staticmethod
    @abstractmethod
    def aggregate(events: QuerySet[AnalyticsEvent], aggregated_at: datetime) -> None:
        """
        Aggregate the given events into the corresponding aggregate model and
        mark the events that were consumed by setting ``aggregated_at``.

        The queryset is already filtered to events of this aggregator's types,
        bounded by the snapshot ID, restricted to ``aggregated_at__isnull=True``
        and annotated with ``event_date`` (UTC date of the event).
        """


class JobSelectionAggregator(EventAggregator):
    """
    Aggregates job_selected events into daily JobSelectionAggregate records.
    selection_count = number of "add" events minus number of "remove" events.
    """

    event_types = [AnalyticsEvent.EventType.JOB_SELECTED]

    @staticmethod
    def aggregate(events: QuerySet[AnalyticsEvent], aggregated_at: datetime) -> None:
        daily_stats = (
            events.annotate(job_id=KT("payload__job_id"))
            .values("job_id", "event_date")
            .annotate(
                selection_count=(
                    Count("id", filter=Q(payload__action="add"))
                    - Count("id", filter=Q(payload__action="remove"))
                ),
            )
        )

        for stat in daily_stats:
            aggregate, _created = JobSelectionAggregate.objects.update_or_create(
                job_id=stat["job_id"],
                date=stat["event_date"],
                defaults={
                    "selection_count": F("selection_count") + stat["selection_count"]
                },
                create_defaults={"selection_count": stat["selection_count"]},
            )
            logger.info("Created or updated aggregate %r", aggregate)

        events.update(aggregated_at=aggregated_at)


class SessionAggregator(EventAggregator):
    """
    Aggregates session_start and session_end events into daily SessionAggregate
    records. Pairs events by session_id to compute duration.

    Sessions whose start and end fall on different UTC days are attributed to
    their start day. Sessions that cannot be paired in this run (for example a
    session_start whose session_end has not arrived yet, or a session_id that
    appears more than twice) are left unmarked so a later run can pick them up
    once the matching event arrives.
    """

    event_types = [
        AnalyticsEvent.EventType.SESSION_START,
        AnalyticsEvent.EventType.SESSION_END,
    ]

    @classmethod
    def aggregate(
        cls, events: QuerySet[AnalyticsEvent], aggregated_at: datetime
    ) -> None:
        consumed_session_ids: list[str] = []
        for session in cls.sessions(events):
            date = session["event_date"]
            duration_seconds = int(
                (session["end_timestamp"] - session["timestamp"]).total_seconds()
            )
            aggregate, _created = SessionAggregate.objects.update_or_create(
                date=date,
                defaults={
                    "total_sessions": F("total_sessions") + 1,
                    "total_duration_seconds": F("total_duration_seconds")
                    + duration_seconds,
                },
                create_defaults={
                    "total_sessions": 1,
                    "total_duration_seconds": duration_seconds,
                },
            )
            consumed_session_ids.append(session["payload__session_id"])
            logger.info(
                "Created or updated aggregate %r with session %r", aggregate, session
            )

        if consumed_session_ids:
            events.filter(payload__session_id__in=consumed_session_ids).update(
                aggregated_at=aggregated_at
            )

    @staticmethod
    def sessions(
        events: QuerySet[AnalyticsEvent],
    ) -> QuerySet[AnalyticsEvent, dict[str, Any]]:
        """
        Converts the query set of session_start and session_end events into a query set of
        valid sessions with `event_date`, `timestamp`, `end_timestamp` and `payload__session_id` keys.
        """
        # First filter on the number of unique session ids.
        # Any session which does not occur exactly twice in the dataset is invalid.
        # For example, a session that was started but not ended, or ended multiple times.
        # This still leaves some invalid data, like sessions that were ended twice or started twice.
        possibly_valid_sessions = (
            events.values("payload__session_id")
            .annotate(num_session_ids=Count("pk"))
            .filter(num_session_ids=2)
            .values("payload__session_id")
        )
        # Next figure out all session end events that may be valid.
        possibly_valid_ends = events.filter(
            event_type=AnalyticsEvent.EventType.SESSION_END,
            payload__session_id__in=possibly_valid_sessions,
        )
        # With the possibly valid ends we can now figure out all the valid pairs of (session_start, session_end):
        # The main condition is: session_id is used exactly twice, there exists an end event with the session id
        # and there exists a start event with the session id. If this is true, we know that the session is valid.
        return (
            events.filter(
                event_type=AnalyticsEvent.EventType.SESSION_START,
            )
            .annotate(
                end_timestamp=Subquery(
                    possibly_valid_ends.filter(
                        payload__session_id=OuterRef("payload__session_id")
                    ).values("timestamp")
                )
            )
            .values("event_date", "timestamp", "end_timestamp", "payload__session_id")
            .filter(end_timestamp__isnull=False)
        )


class ModuleDurationAggregator(EventAggregator):
    """
    Aggregates module duration events.
    """

    event_types = [AnalyticsEvent.EventType.MODULE_DURATION]

    @staticmethod
    def aggregate(events: QuerySet[AnalyticsEvent], aggregated_at: datetime) -> None:
        aggregated_events = (
            events.annotate(
                duration_seconds=Cast(
                    KT("payload__duration_seconds"), output_field=IntegerField()
                )
            )
            .values("event_date", "payload__exercise_type", "payload__unit_id")
            .annotate(
                total_duration_seconds=Sum("duration_seconds"),
                total_sessions=Count("pk"),
            )
        )

        for event in aggregated_events:
            aggregate, _created = ModuleDurationAggregate.objects.update_or_create(
                date=event["event_date"],
                exercise_type=event["payload__exercise_type"],
                unit_id=event["payload__unit_id"],
                defaults={
                    "total_duration_seconds": F("total_duration_seconds")
                    + event["total_duration_seconds"],
                    "total_sessions": F("total_sessions") + event["total_sessions"],
                },
                create_defaults={
                    "total_duration_seconds": event["total_duration_seconds"],
                    "total_sessions": event["total_sessions"],
                },
            )
            logger.info("Created or updated aggregate %r", aggregate)

        events.update(aggregated_at=aggregated_at)


class DropoutAggregator(EventAggregator):
    """
    Aggregates exercise dropout events into daily DropoutAggregate records.
    Groups by (date, exercise_type, unit_id, position, total) and sums dropout counts.
    """

    event_types = [AnalyticsEvent.EventType.EXERCISE_DROPOUT]

    @staticmethod
    def aggregate(events: QuerySet[AnalyticsEvent], aggregated_at: datetime) -> None:
        aggregated_events = events.values(
            "event_date",
            "payload__exercise_type",
            "payload__unit_id",
            "payload__position",
            "payload__total",
        ).annotate(
            dropout_count=Count("pk"),
        )

        for event in aggregated_events:
            aggregate, _created = DropoutAggregate.objects.update_or_create(
                date=event["event_date"],
                exercise_type=event["payload__exercise_type"],
                unit_id=event["payload__unit_id"],
                dropout_position=event["payload__position"],
                total_items=event["payload__total"],
                defaults={
                    "dropout_count": F("dropout_count") + event["dropout_count"],
                },
                create_defaults={
                    "dropout_count": event["dropout_count"],
                },
            )
            logger.info("Created or updated aggregate %r", aggregate)

        events.update(aggregated_at=aggregated_at)


EVENT_AGGREGATORS: list[type[EventAggregator]] = [
    JobSelectionAggregator,
    SessionAggregator,
    ModuleDurationAggregator,
    DropoutAggregator,
]


class Command(BaseCommand):
    """
    Aggregate analytics events into daily summaries.

    Raw events are kept after aggregation; each consumed event is stamped with
    ``aggregated_at`` so subsequent runs only process newly arrived events.
    """

    help = "Aggregate analytics events into daily summaries (raw events are retained)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Run the full aggregation pipeline but roll back all changes.",
        )

    def handle(self, *args, **options) -> None:
        dry_run: bool = options["dry_run"]
        for aggregator_class in EVENT_AGGREGATORS:
            self._aggregate_event_type(aggregator_class, dry_run)

    @transaction.atomic
    def _aggregate_event_type(
        self,
        aggregator_class: type[EventAggregator],
        dry_run: bool,
    ) -> None:
        event_types = aggregator_class.event_types

        # Capture max ID to avoid touching events that arrive during aggregation.
        max_id = AnalyticsEvent.objects.filter(
            event_type__in=event_types,
            aggregated_at__isnull=True,
        ).aggregate(max_id=models.Max("id"))["max_id"]

        if max_id is None:
            self.stdout.write(f"No new {event_types} events to aggregate.")
            return

        events = AnalyticsEvent.objects.filter(
            event_type__in=event_types,
            aggregated_at__isnull=True,
            id__lte=max_id,
        ).annotate(event_date=TruncDate("timestamp", tzinfo=UTC))

        aggregated_at = timezone.now()
        aggregator_class.aggregate(events, aggregated_at)

        marked_count = AnalyticsEvent.objects.filter(
            event_type__in=event_types,
            aggregated_at=aggregated_at,
        ).count()

        if dry_run:
            transaction.set_rollback(True)
            self.stdout.write(
                f"[DRY RUN] Would aggregate and mark {marked_count} {event_types} events."
            )
        else:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Aggregated and marked {marked_count} {event_types} events."
                )
            )
