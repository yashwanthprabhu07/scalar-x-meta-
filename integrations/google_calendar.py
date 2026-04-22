# ============================================================
# integrations/google_calendar.py
#
# Real Google Calendar implementation that matches the mock
# CalendarApp interface from mock_apps.py. Drop-in replacement.
#
# Usage:
#   from integrations.google_calendar import RealCalendarApp
#   cal = RealCalendarApp()
#   cal.book_meeting(...)
#
# Requires:
#   credentials.json (OAuth client secrets)
#   token.json       (created on first use via integrations/test_calendar_auth.py)
#
# Both files are in .gitignore.
# ============================================================

import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"
DEFAULT_TIMEZONE = "Asia/Kolkata"  # IST (for the hackathon)


# ─────────────────────────────────────────────
# AUTHENTICATION HELPER
# ─────────────────────────────────────────────
def _get_service():
    """Return an authenticated Calendar API service. Opens browser on first run."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_FILE} — download from Google Cloud Console"
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


# ─────────────────────────────────────────────
# DATE/TIME HELPERS
# ─────────────────────────────────────────────
def _combine_date_time(date_str: str, time_str: str) -> datetime:
    """
    Combine 'YYYY-MM-DD' and 'HH:MM' into a timezone-naive datetime.
    Caller adds timezone afterward.
    """
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")


def _format_iso(dt: datetime, tz: str = DEFAULT_TIMEZONE) -> str:
    """Return a Google Calendar–friendly ISO datetime string with timezone."""
    # For Google Calendar's timed events, we use local-time ISO + timeZone field separately
    return dt.isoformat()


# ─────────────────────────────────────────────
# THE REAL CALENDAR APP
# ─────────────────────────────────────────────
class RealCalendarApp:
    """
    Real Google Calendar implementation matching the mock CalendarApp interface.

    Methods (same signatures as mock):
      - list_meetings(date=None) -> list[dict]
      - check_conflicts(date, time, attendees) -> dict
      - book_meeting(title, attendees, date, time, duration_mins=60) -> dict
    """

    def __init__(self, calendar_id: str = "primary", timezone_name: str = DEFAULT_TIMEZONE):
        self.service = _get_service()
        self.calendar_id = calendar_id
        self.timezone_name = timezone_name

    # ── list_meetings ───────────────────────────────────────
    def list_meetings(self, date: str = None) -> List[Dict[str, Any]]:
        """
        List meetings. If date is None, returns upcoming 10 events.
        If date is 'YYYY-MM-DD', returns events on that specific day.
        """
        try:
            if date:
                # Events for the specific day
                day_start = datetime.strptime(date, "%Y-%m-%d")
                day_end = day_start + timedelta(days=1)
                time_min = day_start.isoformat() + "Z"
                time_max = day_end.isoformat() + "Z"
            else:
                time_min = datetime.now(timezone.utc).isoformat()
                time_max = None

            params = {
                "calendarId": self.calendar_id,
                "timeMin": time_min,
                "maxResults": 20,
                "singleEvents": True,
                "orderBy": "startTime",
            }
            if time_max:
                params["timeMax"] = time_max

            events_result = self.service.events().list(**params).execute()
            events = events_result.get("items", [])

            # Shape the response to match the mock app's format
            return [
                {
                    "id": e.get("id"),
                    "title": e.get("summary", "(no title)"),
                    "start": e["start"].get("dateTime", e["start"].get("date")),
                    "end": e["end"].get("dateTime", e["end"].get("date")),
                    "attendees": [a.get("email") for a in e.get("attendees", [])],
                    "link": e.get("htmlLink"),
                }
                for e in events
            ]

        except HttpError as e:
            return [{"error": f"Google Calendar API error: {e}"}]

    # ── check_conflicts ─────────────────────────────────────
    def check_conflicts(self, date: str, time: str, attendees: List[str]) -> Dict[str, Any]:
        """
        Check if the primary calendar has conflicts at the given date+time.
        For simplicity, we check only the primary calendar (not each attendee's).
        """
        try:
            start_dt = _combine_date_time(date, time)
            end_dt = start_dt + timedelta(hours=1)

            freebusy_req = {
                "timeMin": start_dt.isoformat() + "Z",
                "timeMax": end_dt.isoformat() + "Z",
                "items": [{"id": self.calendar_id}],
            }
            result = self.service.freebusy().query(body=freebusy_req).execute()
            busy_slots = result["calendars"][self.calendar_id].get("busy", [])

            return {
                "has_conflict": len(busy_slots) > 0,
                "conflicts": busy_slots,
                "checked_at": f"{date} {time}",
            }

        except HttpError as e:
            return {"error": f"Google Calendar API error: {e}"}

    # ── book_meeting ────────────────────────────────────────
    def book_meeting(
        self,
        title: str,
        attendees: List[str],
        date: str,
        time: str,
        duration_mins: int = 60,
    ) -> Dict[str, Any]:
        """
        Book a real meeting on the user's primary Google Calendar.
        Returns {meeting_id, status, link} on success; {error} on failure.
        """
        try:
            start_dt = _combine_date_time(date, time)
            end_dt = start_dt + timedelta(minutes=duration_mins)

            event_body = {
                "summary": title,
                "description": (
                    f"Meeting auto-booked by the Enterprise Workflow Agent "
                    f"(hackathon demo). Attendees: {', '.join(attendees)}"
                ),
                "start": {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": self.timezone_name,
                },
                "end": {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": self.timezone_name,
                },
                "attendees": [{"email": a} for a in attendees if "@" in a],
                "reminders": {"useDefault": True},
            }

            created = self.service.events().insert(
                calendarId=self.calendar_id,
                body=event_body,
                sendUpdates="none",  # don't email attendees (fake addresses in demo)
            ).execute()

            return {
                "meeting_id": created.get("id"),
                "status": "booked",
                "link": created.get("htmlLink"),
                "title": title,
                "date": date,
                "time": time,
                "duration_mins": duration_mins,
            }

        except HttpError as e:
            return {"error": f"Google Calendar API error: {e}"}


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("── RealCalendarApp smoke test ──")
    cal = RealCalendarApp()

    print("\nUpcoming meetings:")
    for m in cal.list_meetings()[:5]:
        print(f"  {m['start']} — {m['title']}")

    print("\nChecking conflicts for 2026-04-23 10:00 ...")
    print(cal.check_conflicts("2026-04-23", "10:00", ["test@example.com"]))

    print("\nBooking test meeting ...")
    result = cal.book_meeting(
        title="Hackathon Test Meeting — SAFE TO DELETE",
        attendees=["rajesh.kumar@acmecorp.com"],
        date="2026-04-23",
        time="10:00",
        duration_mins=30,
    )
    print("Result:", result)
    if result.get("link"):
        print(f"\n→ Open this to see the meeting: {result['link']}")