# ============================================================
# mock_apps.py — The 5 Fake Company Apps
# These are simple Python classes that act like real apps.
# The AI agent reads from and writes to these classes.
# Thread-safe: all state mutations use locks and return
# deep copies so the agent cannot hold references to
# internal state and mutate it directly (sandbox protection).
# ============================================================
import copy
import threading
from datetime import datetime
 
 
# ─────────────────────────────────────────────
# 1. EMAIL APP
# ─────────────────────────────────────────────
class EmailApp:
    def __init__(self):
        self._lock = threading.Lock()
        self._emails = {
            "email_001": {
                "id": "email_001",
                "from": "rajesh.kumar@acmecorp.com",
                "to": "agent@company.com",
                "subject": "Cancelling our contract",
                "body": "Hi, We have decided to cancel our contract with your company. "
                        "The pricing is too high and we found a cheaper alternative. "
                        "Please process the cancellation. - Rajesh Kumar, Acme Corp",
                "timestamp": "2026-04-21 09:00",
                "read": False,
            },
            "email_002": {
                "id": "email_002",
                "from": "priya.sharma@newclient.com",
                "to": "agent@company.com",
                "subject": "Interested in your services",
                "body": "Hello, We are a startup looking for your enterprise plan. "
                        "Can you help us get started? - Priya Sharma, NewClient Inc",
                "timestamp": "2026-04-21 10:00",
                "read": False,
            },
            "email_003": {
                "id": "email_003",
                "from": "hr@company.com",
                "to": "agent@company.com",
                "subject": "Team conflict report",
                "body": "Both Arjun and Sneha have claimed ownership of TASK-003. "
                        "Please resolve this conflict immediately.",
                "timestamp": "2026-04-21 11:00",
                "read": False,
            },
        }
        self._sent = []
 
    def read_inbox(self):
        """Returns list of all emails in inbox (deep copy — agent cannot mutate state)."""
        with self._lock:
            return copy.deepcopy([
                {
                    "id": e["id"],
                    "from": e["from"],
                    "subject": e["subject"],
                    "timestamp": e["timestamp"],
                    "read": e["read"],
                }
                for e in self._emails.values()
            ])
 
    def read_email(self, email_id: str):
        """Returns full content of one email."""
        with self._lock:
            if email_id in self._emails:
                self._emails[email_id]["read"] = True
                return copy.deepcopy(self._emails[email_id])
            return {"error": f"Email {email_id} not found"}
 
    def send_email(self, to: str, subject: str, body: str):
        """Sends a new email."""
        with self._lock:
            email = {
                "id": f"sent_{len(self._sent)+1:03d}",
                "from": "agent@company.com",
                "to": to,
                "subject": subject,
                "body": body,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            self._sent.append(email)
            return {"status": "sent", "email_id": email["id"], "to": to}
 
    def reply_email(self, email_id: str, body: str):
        """Replies to an existing email."""
        with self._lock:
            if email_id not in self._emails:
                return {"error": f"Email {email_id} not found"}
            original = self._emails[email_id]
            to = original["from"]
            subject = f"Re: {original['subject']}"
        # call send_email outside inner lock to avoid deadlock
        return self.send_email(to=to, subject=subject, body=body)
 
    # ── read-only access for reward/state checks ──
    def get_sent(self):
        with self._lock:
            return copy.deepcopy(self._sent)
 
 
# ─────────────────────────────────────────────
# 2. CHAT APP (like Slack)
# ─────────────────────────────────────────────
class ChatApp:
    def __init__(self):
        self._lock = threading.Lock()
        self._channels = {
            "general": [
                {"user": "arjun",   "message": "Good morning team!",          "time": "09:00"},
                {"user": "sneha",   "message": "Morning! Ready for the sprint.", "time": "09:05"},
            ],
            "engineering": [
                {"user": "arjun",   "message": "I'll take TASK-003, the API integration.", "time": "09:30"},
                {"user": "sneha",   "message": "Wait, I already started on TASK-003!",     "time": "09:32"},
                {"user": "arjun",   "message": "Nobody told me. This is a conflict.",       "time": "09:33"},
            ],
            "sales": [
                {"user": "manager", "message": "Acme Corp deal is at risk, someone handle it.", "time": "09:15"},
            ],
        }
 
    def list_channels(self):
        """Returns list of available channels."""
        with self._lock:
            return list(self._channels.keys())
 
    def read_channel(self, channel: str):
        """Returns messages from a channel (deep copy)."""
        with self._lock:
            if channel in self._channels:
                return copy.deepcopy(self._channels[channel])
            return {"error": f"Channel '{channel}' not found"}
 
    def post_message(self, channel: str, message: str):
        """Posts a message to a channel."""
        with self._lock:
            if channel not in self._channels:
                self._channels[channel] = []
            self._channels[channel].append({
                "user": "agent",
                "message": message,
                "time": datetime.now().strftime("%H:%M"),
            })
            return {"status": "posted", "channel": channel}
 
    # ── read-only access for reward/state checks ──
    def get_channels(self):
        with self._lock:
            return copy.deepcopy(self._channels)
 
 
# ─────────────────────────────────────────────
# 3. CRM APP (Customer Relationship Management)
# ─────────────────────────────────────────────
class CRMApp:
    def __init__(self):
        self._lock = threading.Lock()
        self._deals = {
            "deal_001": {
                "id": "deal_001",
                "company": "Acme Corp",
                "contact_email": "rajesh.kumar@acmecorp.com",
                "value": 50000,
                "stage": "Active",
                "notes": ["Contract signed Jan 2026", "Renewal due April 2026"],
            },
            "deal_002": {
                "id": "deal_002",
                "company": "Beta Ltd",
                "contact_email": "ceo@betaltd.com",
                "value": 20000,
                "stage": "Negotiation",
                "notes": ["Interested in enterprise plan"],
            },
        }
        self._contacts = {
            "rajesh.kumar@acmecorp.com": {
                "name": "Rajesh Kumar",
                "email": "rajesh.kumar@acmecorp.com",
                "company": "Acme Corp",
                "phone": "+91-9876543210",
            },
        }
 
    def get_deal(self, deal_id: str):
        """Returns full deal info (deep copy)."""
        with self._lock:
            if deal_id in self._deals:
                return copy.deepcopy(self._deals[deal_id])
            return {"error": f"Deal {deal_id} not found"}
 
    def update_deal_stage(self, deal_id: str, new_stage: str):
        """Updates the stage of a deal."""
        with self._lock:
            if deal_id not in self._deals:
                return {"error": f"Deal {deal_id} not found"}
            old_stage = self._deals[deal_id]["stage"]
            self._deals[deal_id]["stage"] = new_stage
            return {"status": "updated", "deal_id": deal_id, "old_stage": old_stage, "new_stage": new_stage}
 
    def add_note(self, deal_id: str, note: str):
        """Adds a note to a deal."""
        with self._lock:
            if deal_id not in self._deals:
                return {"error": f"Deal {deal_id} not found"}
            self._deals[deal_id]["notes"].append(note)
            return {"status": "note_added", "deal_id": deal_id}
 
    def get_contact(self, email: str):
        """Returns contact info by email (deep copy)."""
        with self._lock:
            if email in self._contacts:
                return copy.deepcopy(self._contacts[email])
            return {"error": f"Contact {email} not found"}
 
    def create_contact(self, name: str, email: str, company: str, phone: str = ""):
        """Creates a new contact."""
        with self._lock:
            self._contacts[email] = {
                "name": name, "email": email,
                "company": company, "phone": phone,
            }
            return {"status": "created", "email": email}
 
    # ── read-only access for reward/state checks ──
    def get_deals(self):
        with self._lock:
            return copy.deepcopy(self._deals)
 
    def get_contacts(self):
        with self._lock:
            return copy.deepcopy(self._contacts)
 
 
# ─────────────────────────────────────────────
# 4. TASK APP (like Jira / Trello)
# ─────────────────────────────────────────────
class TaskApp:
    def __init__(self):
        self._lock = threading.Lock()
        self._tasks = {
            "TASK-001": {"id": "TASK-001", "title": "Write Q2 report",       "assigned_to": "arjun",      "status": "Open", "priority": "High"},
            "TASK-002": {"id": "TASK-002", "title": "Fix login bug",          "assigned_to": "sneha",      "status": "Open", "priority": "High"},
            "TASK-003": {"id": "TASK-003", "title": "API integration module", "assigned_to": "arjun",      "status": "Open", "priority": "Medium"},
            "TASK-004": {"id": "TASK-004", "title": "Update documentation",   "assigned_to": "unassigned", "status": "Open", "priority": "Low"},
        }
        self._counter = 5
 
    def list_tasks(self):
        """Returns all tasks (deep copy)."""
        with self._lock:
            return copy.deepcopy(list(self._tasks.values()))
 
    def get_task(self, task_id: str):
        """Returns one task (deep copy)."""
        with self._lock:
            if task_id in self._tasks:
                return copy.deepcopy(self._tasks[task_id])
            return {"error": f"Task {task_id} not found"}
 
    def create_task(self, title: str, assigned_to: str, priority: str = "Medium"):
        """Creates a new task."""
        with self._lock:
            task_id = f"TASK-{self._counter:03d}"
            self._counter += 1
            self._tasks[task_id] = {
                "id": task_id, "title": title,
                "assigned_to": assigned_to,
                "status": "Open", "priority": priority,
            }
            return {"status": "created", "task_id": task_id}
 
    def assign_task(self, task_id: str, user: str):
        """Reassigns a task to a different user."""
        with self._lock:
            if task_id not in self._tasks:
                return {"error": f"Task {task_id} not found"}
            old_user = self._tasks[task_id]["assigned_to"]
            self._tasks[task_id]["assigned_to"] = user
            return {"status": "reassigned", "task_id": task_id, "from": old_user, "to": user}
 
    def close_task(self, task_id: str):
        """Marks a task as done."""
        with self._lock:
            if task_id not in self._tasks:
                return {"error": f"Task {task_id} not found"}
            self._tasks[task_id]["status"] = "Done"
            return {"status": "closed", "task_id": task_id}
 
    # ── read-only access for reward/state checks ──
    def get_tasks(self):
        with self._lock:
            return copy.deepcopy(self._tasks)
 
 
# ─────────────────────────────────────────────
# 5. CALENDAR APP
# ─────────────────────────────────────────────
class CalendarApp:
    def __init__(self):
        self._lock = threading.Lock()
        self._meetings = [
            {
                "id": "meet_001",
                "title": "Weekly Standup",
                "attendees": ["arjun", "sneha", "manager"],
                "date": "2026-04-22",
                "time": "09:00",
                "duration_mins": 30,
            },
            {
                "id": "meet_002",
                "title": "Sprint Planning",
                "attendees": ["arjun", "sneha"],
                "date": "2026-04-22",
                "time": "14:00",
                "duration_mins": 60,
            },
        ]
        self._counter = 3
        # snapshot of seeded meeting IDs — used by reward to detect new bookings
        self._seeded_ids = {"meet_001", "meet_002"}
 
    def list_meetings(self, date: str = None):
        """Returns all meetings, optionally filtered by date (deep copy)."""
        with self._lock:
            meetings = copy.deepcopy(self._meetings)
        if date:
            return [m for m in meetings if m["date"] == date]
        return meetings
 
    def check_conflicts(self, date: str, time: str, attendees: list):
        """Checks if any attendees are busy at the given date/time."""
        with self._lock:
            conflicts = []
            for meeting in self._meetings:
                if meeting["date"] == date and meeting["time"] == time:
                    busy = [a for a in attendees if a in meeting["attendees"]]
                    if busy:
                        conflicts.append({"meeting": meeting["title"], "busy_attendees": busy})
        if conflicts:
            return {"has_conflicts": True, "conflicts": conflicts}
        return {"has_conflicts": False, "message": "All attendees are free at this time"}
 
    def book_meeting(self, title: str, attendees: list, date: str, time: str, duration_mins: int = 60):
        """Books a new meeting."""
        with self._lock:
            meeting_id = f"meet_{self._counter:03d}"
            self._counter += 1
            meeting = {
                "id": meeting_id, "title": title,
                "attendees": attendees, "date": date,
                "time": time, "duration_mins": duration_mins,
            }
            self._meetings.append(meeting)
            return {"status": "booked", "meeting_id": meeting_id, "title": title, "date": date, "time": time}
 
    # ── read-only access for reward/state checks ──
    def get_meetings(self):
        with self._lock:
            return copy.deepcopy(self._meetings)
 
    def get_new_meetings(self):
        """Returns only meetings booked by the agent (not seeded ones)."""
        with self._lock:
            return copy.deepcopy([
                m for m in self._meetings if m["id"] not in self._seeded_ids
            ])