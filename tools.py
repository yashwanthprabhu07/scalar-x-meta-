# ============================================================
# tools.py — Tool Definitions for the AI Agent
# These tell the AI what functions it can call and what
# arguments each function needs.
# ============================================================

ALL_TOOLS = [
    # ── EMAIL TOOLS ──────────────────────────────────────────
    {
        "name": "read_inbox",
        "description": "Read all emails in the inbox. Use this first to see what needs attention.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_email",
        "description": "Read the full content of a specific email by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string", "description": "The email ID e.g. email_001"}
            },
            "required": ["email_id"],
        },
    },
    {
        "name": "send_email",
        "description": "Send a new email to someone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to":      {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body":    {"type": "string", "description": "Email body content"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "reply_email",
        "description": "Reply to an existing email.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string", "description": "The ID of the email to reply to"},
                "body":     {"type": "string", "description": "Your reply message"},
            },
            "required": ["email_id", "body"],
        },
    },

    # ── CHAT TOOLS ───────────────────────────────────────────
    {
        "name": "list_channels",
        "description": "List all available Slack-like chat channels.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_channel",
        "description": "Read messages from a specific chat channel.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Channel name e.g. engineering, sales, general"}
            },
            "required": ["channel"],
        },
    },
    {
        "name": "post_message",
        "description": "Post a message to a chat channel to notify the team.",
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "description": "Channel name to post in"},
                "message": {"type": "string", "description": "The message to post"},
            },
            "required": ["channel", "message"],
        },
    },

    # ── CRM TOOLS ────────────────────────────────────────────
    {
        "name": "get_deal",
        "description": "Get full details of a CRM deal by deal ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_id": {"type": "string", "description": "Deal ID e.g. deal_001"}
            },
            "required": ["deal_id"],
        },
    },
    {
        "name": "update_deal_stage",
        "description": "Update the stage of a deal. Stages: Active, Negotiation, Won, Lost, At Risk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_id":   {"type": "string", "description": "Deal ID"},
                "new_stage": {"type": "string", "description": "New stage: Active, Negotiation, Won, Lost, At Risk"},
            },
            "required": ["deal_id", "new_stage"],
        },
    },
    {
        "name": "add_note",
        "description": "Add a note to a CRM deal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "deal_id": {"type": "string", "description": "Deal ID"},
                "note":    {"type": "string", "description": "Note to add"},
            },
            "required": ["deal_id", "note"],
        },
    },
    {
        "name": "get_contact",
        "description": "Get contact info from CRM by email address.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Contact email address"}
            },
            "required": ["email"],
        },
    },
    {
        "name": "create_contact",
        "description": "Create a new contact in the CRM.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name":    {"type": "string", "description": "Full name"},
                "email":   {"type": "string", "description": "Email address"},
                "company": {"type": "string", "description": "Company name"},
                "phone":   {"type": "string", "description": "Phone number (optional)"},
            },
            "required": ["name", "email", "company"],
        },
    },

    # ── TASK TOOLS ───────────────────────────────────────────
    {
        "name": "list_tasks",
        "description": "List all tasks in the task manager.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_task",
        "description": "Get details of a specific task by task ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID e.g. TASK-003"}
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "create_task",
        "description": "Create a new task and assign it to someone.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title":       {"type": "string", "description": "Task title"},
                "assigned_to": {"type": "string", "description": "Username to assign to"},
                "priority":    {"type": "string", "description": "Priority: Low, Medium, High"},
            },
            "required": ["title", "assigned_to"],
        },
    },
    {
        "name": "assign_task",
        "description": "Reassign an existing task to a different team member.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to reassign"},
                "user":    {"type": "string", "description": "Username to assign to"},
            },
            "required": ["task_id", "user"],
        },
    },
    {
        "name": "close_task",
        "description": "Mark a task as done/completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task ID to close"}
            },
            "required": ["task_id"],
        },
    },

    # ── CALENDAR TOOLS ───────────────────────────────────────
    {
        "name": "list_meetings",
        "description": "List all upcoming meetings, optionally filtered by date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Filter by date YYYY-MM-DD (optional)"}
            },
            "required": [],
        },
    },
    {
        "name": "check_conflicts",
        "description": "Check if attendees are free at a specific date and time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date":      {"type": "string", "description": "Date YYYY-MM-DD"},
                "time":      {"type": "string", "description": "Time HH:MM"},
                "attendees": {"type": "array",  "items": {"type": "string"}, "description": "List of usernames"},
            },
            "required": ["date", "time", "attendees"],
        },
    },
    {
        "name": "book_meeting",
        "description": "Book a new meeting in the calendar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title":        {"type": "string", "description": "Meeting title"},
                "attendees":    {"type": "array",  "items": {"type": "string"}, "description": "List of attendees"},
                "date":         {"type": "string", "description": "Date YYYY-MM-DD"},
                "time":         {"type": "string", "description": "Time HH:MM"},
                "duration_mins":{"type": "integer","description": "Duration in minutes (default 60)"},
            },
            "required": ["title", "attendees", "date", "time"],
        },
    },
]
