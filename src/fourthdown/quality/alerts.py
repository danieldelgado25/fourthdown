"""Where a failed check goes: the log always, and a chat webhook when one is configured.

``FOURTHDOWN_ALERT_WEBHOOK`` takes a Slack or Discord incoming-webhook URL (the payload
carries both ``text`` and ``content``, so either accepts it). Unset, alerts are log lines
and failed checks in the Dagster UI. The URL is a credential: it is read from the
environment and never logged.
"""

from __future__ import annotations

import logging
import os

import requests

LOGGER = logging.getLogger(__name__)

WEBHOOK_ENV = "FOURTHDOWN_ALERT_WEBHOOK"
TIMEOUT = 10


def send(title: str, body: str) -> bool:
    """Report a failure; True when a webhook accepted it. Never raises."""
    LOGGER.error("ALERT %s\n%s", title, body)
    url = os.environ.get(WEBHOOK_ENV, "").strip()
    if not url:
        return False
    text = f"fourthdown: {title}\n{body}"[:1900]  # Discord's limit is 2000 characters
    try:
        response = requests.post(url, json={"text": text, "content": text}, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as error:
        # The exception text can include the URL; name only its type.
        LOGGER.warning("the alert webhook did not accept the alert (%s)", type(error).__name__)
        return False
    return True
