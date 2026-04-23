# ============================================================
# mock_apps.py — The 5 Fake Company Apps
# These are simple Python classes that act like real apps.
# The AI agent reads from and writes to these classes.
# ============================================================

from datetime import datetime

# ─────────────────────────────────────────────
# 1. EMAIL APP
# ─────────────────────────────────────────────
class EmailApp:
    def __init__(self):
        # Pre-loaded fake emails in the inbox
        self.emails = {
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
        self.sent = []

    def read_inbox(self):
        """Returns list of all emails in inbox."""
        return [
            {
                "id": e["id"],
                "from": e["from"],
                "subject": e["subject"],
                "timestamp": e["timestamp"],
                "read": e["read"],
            }
            for e in self.emails.values()
        ]

    def read_email(self, email_id: str):
        """Returns full content of one email."""
        if email_id in self.emails:
            self.emails[email_id]["read"] = True
            return self.emails[email_id]
        return {"error": f"Email {email_id} not found"}

    def send_email(self, to: str, subject: str, body: str):
        """Sends a new email."""
        email = {
            "id": f"sent_{len(self.sent)+1:03d}",
            "from": "agent@company.com",
            "to": to,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self.sent.append(email)
        return {"status": "sent", "email_id": email["id"], "to": to}

    def reply_email(self, email_id: str, body: str):
        """Replies to an existing email."""
        if email_id not in self.emails:
            return {"error": f"Email {email_id} not found"}
        original = self.emails[email_id]
        return self.send_email(
            to=original["from"],
            subject=f"Re: {original['subject']}",
            body=body,
        )


# ─────────────────────────────────────────────
# 2. CHAT APP (like Slack)
# ─────────────────────────────────────────────
class ChatApp:
    def __init__(self):
        self.channels = {
            "general": [
                {"user": "arjun", "message": "Good morning team!", "time": "09:00"},
                {"user": "sneha", "message": "Morning! Ready for the sprint.", "time": "09:05"},
            ],
            "engineering": [
                {"user": "arjun",  "message": "I'll take TASK-003, the API integration.", "time": "09:30"},
                {"user": "sneha",  "message": "Wait, I already started on TASK-003!", "time": "09:32"},
                {"user": "arjun",  "message": "Nobody told me. This is a conflict.", "time": "09:33"},
            ],
            "sales": [
                {"user": "manager", "message": "Acme Corp deal is at risk, someone handle it.", "time": "09:15"},
            ],
        }

    def list_channels(self):
        """Returns list of available channels."""
        return list(self.channels.keys())

    def read_channel(self, channel: str):
        """Returns messages from a channel."""
        if channel in self.channels:
            return self.channels[channel]
        return {"error": f"Channel '{channel}' not found"}

    def post_message(self, channel: str, message: str):
        """Posts a message to a channel."""
        if channel not in self.channels:
            self.channels[channel] = []
        self.channels[channel].append({
            "user": "agent",
            "message": message,
            "time": datetime.now().strftime("%H:%M"),
        })
        return {"status": "posted", "channel": channel}


# ─────────────────────────────────────────────
# 3. CRM APP (Customer Relationship Management)
# ─────────────────────────────────────────────
class CRMApp:
    def __init__(self):
        self.deals = {
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
        self.contacts = {
            "rajesh.kumar@acmecorp.com": {
                "name": "Rajesh Kumar",
                "email": "rajesh.kumar@acmecorp.com",
                "company": "Acme Corp",
                "phone": "+91-9876543210",
            },
        }

    def get_deal(self, deal_id: str):
        """Returns full deal info."""
        if deal_id in self.deals:
            return self.deals[deal_id]
        return {"error": f"Deal {deal_id} not found"}

    def update_deal_stage(self, deal_id: str, new_stage: str):
        """Updates the stage of a deal (e.g. Active → Negotiation → Won → Lost)."""
        if deal_id not in self.deals:
            return {"error": f"Deal {deal_id} not found"}
        old_stage = self.deals[deal_id]["stage"]
        self.deals[deal_id]["stage"] = new_stage
        return {"status": "updated", "deal_id": deal_id, "old_stage": old_stage, "new_stage": new_stage}

    def add_note(self, deal_id: str, note: str):
        """Adds a note to a deal."""
        if deal_id not in self.deals:
            return {"error": f"Deal {deal_id} not found"}
        self.deals[deal_id]["notes"].append(note)
        return {"status": "note_added", "deal_id": deal_id}

    def get_contact(self, email: str):
        """Returns contact info by email."""
        if email in self.contacts:
            return self.contacts[email]
        return {"error": f"Contact {email} not found"}

    def create_contact(self, name: str, email: str, company: str, phone: str = ""):
        """Creates a new contact."""
        self.contacts[email] = {
            "name": name, "email": email,
            "company": company, "phone": phone,
        }
        return {"status": "created", "email": email}


# ─────────────────────────────────────────────
# 4. TASK APP (like Jira / Trello)
# ─────────────────────────────────────────────
class TaskApp:
    def __init__(self):
        self.tasks = {
            "TASK-001": {"id": "TASK-001", "title": "Write Q2 report",       "assigned_to": "arjun",  "status": "Open",   "priority": "High"},
            "TASK-002": {"id": "TASK-002", "title": "Fix login bug",          "assigned_to": "sneha",  "status": "Open",   "priority": "High"},
            "TASK-003": {"id": "TASK-003", "title": "API integration module", "assigned_to": "arjun",  "status": "Open",   "priority": "Medium"},
            "TASK-004": {"id": "TASK-004", "title": "Update documentation",   "assigned_to": "unassigned", "status": "Open", "priority": "Low"},
        }
        self._counter = 5

    def list_tasks(self):
        """Returns all tasks."""
        return list(self.tasks.values())

    def get_task(self, task_id: str):
        """Returns one task."""
        if task_id in self.tasks:
            return self.tasks[task_id]
        return {"error": f"Task {task_id} not found"}

    def create_task(self, title: str, assigned_to: str, priority: str = "Medium"):
        """Creates a new task."""
        task_id = f"TASK-{self._counter:03d}"
        self._counter += 1
        self.tasks[task_id] = {
            "id": task_id, "title": title,
            "assigned_to": assigned_to,
            "status": "Open", "priority": priority,
        }
        return {"status": "created", "task_id": task_id}

    def assign_task(self, task_id: str, user: str):
        """Reassigns a task to a different user."""
        if task_id not in self.tasks:
            return {"error": f"Task {task_id} not found"}
        old_user = self.tasks[task_id]["assigned_to"]
        self.tasks[task_id]["assigned_to"] = user
        return {"status": "reassigned", "task_id": task_id, "from": old_user, "to": user}

    def close_task(self, task_id: str):
        """Marks a task as done."""
        if task_id not in self.tasks:
            return {"error": f"Task {task_id} not found"}
        self.tasks[task_id]["status"] = "Done"
        return {"status": "closed", "task_id": task_id}


# ─────────────────────────────────────────────
# 5. CALENDAR APP
# ─────────────────────────────────────────────
class CalendarApp:
    def __init__(self):
        self.meetings = [
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

    def list_meetings(self, date: str = None):
        """Returns all meetings, optionally filtered by date."""
        if date:
            return [m for m in self.meetings if m["date"] == date]
        return self.meetings

    def check_conflicts(self, date: str, time: str, attendees: list):
        """Checks if any attendees are busy at the given date/time."""
        conflicts = []
        for meeting in self.meetings:
            if meeting["date"] == date and meeting["time"] == time:
                busy = [a for a in attendees if a in meeting["attendees"]]
                if busy:
                    conflicts.append({"meeting": meeting["title"], "busy_attendees": busy})
        if conflicts:
            return {"has_conflicts": True, "conflicts": conflicts}
        return {"has_conflicts": False, "message": "All attendees are free at this time"}

    def book_meeting(self, title: str, attendees: list, date: str, time: str, duration_mins: int = 60):
        """Books a new meeting."""
        meeting_id = f"meet_{self._counter:03d}"
        self._counter += 1
        meeting = {
            "id": meeting_id, "title": title,
            "attendees": attendees, "date": date,
            "time": time, "duration_mins": duration_mins,
        }
        self.meetings.append(meeting)
        return {"status": "booked", "meeting_id": meeting_id, "title": title, "date": date, "time": time}
