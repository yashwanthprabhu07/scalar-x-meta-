# ============================================================
# integrations/test_calendar_auth.py
#
# Standalone script to verify Google Calendar OAuth works.
#
# First run: opens a browser asking you to sign in and grant
# calendar access. Saves an auth token to token.json so future
# runs don't need the browser.
#
# Then lists your next 10 calendar events as a smoke test.
# ============================================================

import os
import pickle
from datetime import datetime, timezone, timedelta

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"


def get_calendar_service():
    """Returns an authenticated Google Calendar API client."""
    creds = None

    # Load previously-saved token if it exists
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # If no valid creds, do the OAuth dance
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_FILE} — download from Google Cloud Console"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            # Runs a tiny local server to catch the OAuth redirect
            creds = flow.run_local_server(port=0)

        # Save for next time
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def list_upcoming_events(num_results: int = 10):
    """Print the next N events on the user's primary calendar."""
    service = get_calendar_service()

    now = datetime.now(timezone.utc).isoformat()
    print(f"── Fetching next {num_results} events from primary calendar ──")

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=num_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = events_result.get("items", [])
    if not events:
        print("  (no upcoming events)")
        return

    for idx, event in enumerate(events, 1):
        start = event["start"].get("dateTime", event["start"].get("date"))
        summary = event.get("summary", "(no title)")
        print(f"  [{idx}] {start} — {summary}")


if __name__ == "__main__":
    list_upcoming_events()