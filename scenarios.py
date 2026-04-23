# ============================================================
# scenarios.py â€” The Task Scenarios
# Each scenario is a dict that tells the agent what to do,
# defines what "success" looks like as required tool calls,
# AND includes a state-based check that inspects the final
# app state to verify real outcomes (not just tool calls).
# ============================================================


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STATE-BASED SUCCESS CHECKS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def check_deal_rescue(apps: dict) -> tuple:
    """Verify the Acme deal was actually rescued."""
    reasons = []
    passed = True

    crm = apps["crm"]
    calendar = apps["calendar"]
    email = apps["email"]
    chat = apps["chat"]

    acme_deal = None
    for deal in crm.deals.values():
        if "acme" in deal.get("company", "").lower():
            acme_deal = deal
            break

    if acme_deal is None:
        reasons.append("âŒ No Acme deal found in CRM")
        passed = False
    else:
        stage = acme_deal.get("stage", "")
        if stage in ("Negotiation", "At Risk"):
            reasons.append(f"âœ… Acme deal stage updated to '{stage}'")
        else:
            reasons.append(f"âŒ Acme deal stage is still '{stage}', expected Negotiation or At Risk")
            passed = False

        notes = acme_deal.get("notes", [])
        if len(notes) > 2:
            reasons.append(f"âœ… New note added to Acme deal ({len(notes)} notes total)")
        else:
            reasons.append(f"âŒ No new note added to Acme deal (still {len(notes)} notes)")
            passed = False

    follow_up_meetings = [
        m for m in calendar.meetings
        if m["id"] not in ("meet_001", "meet_002")
    ]
    if len(follow_up_meetings) > 0:
        reasons.append(f"âœ… Follow-up meeting booked ({len(follow_up_meetings)} new meeting(s))")
    else:
        reasons.append("âŒ No new follow-up meeting was booked")
        passed = False

    acme_replies = [
        e for e in email.sent
        if "acme" in e.get("to", "").lower() or "rajesh" in e.get("to", "").lower()
    ]
    if len(acme_replies) > 0:
        reasons.append(f"âœ… Reply email sent to Acme ({len(acme_replies)} email(s))")
    else:
        reasons.append("âŒ No reply email was sent to the Acme client")
        passed = False

    sales_messages = chat.channels.get("sales", [])
    agent_sales_msgs = [m for m in sales_messages if m.get("user") == "agent"]
    if len(agent_sales_msgs) > 0:
        reasons.append(f"âœ… Agent posted in sales channel ({len(agent_sales_msgs)} message(s))")
    else:
        reasons.append("âŒ No message posted by agent in sales channel")
        passed = False

    return passed, reasons


def check_team_conflict(apps: dict) -> tuple:
    """Verify the task conflict was actually resolved."""
    reasons = []
    passed = True

    tasks = apps["tasks"]
    calendar = apps["calendar"]
    chat = apps["chat"]

    task_003 = tasks.tasks.get("TASK-003")
    if task_003 is None:
        reasons.append("âŒ TASK-003 not found")
        passed = False
    else:
        assignee = task_003.get("assigned_to", "").lower()
        if "sneha" in assignee:
            reasons.append("âœ… TASK-003 reassigned to Sneha")
        else:
            reasons.append(f"âŒ TASK-003 assigned to '{assignee}', expected Sneha")
            passed = False

    task_004 = tasks.tasks.get("TASK-004")
    if task_004 is None:
        reasons.append("âŒ TASK-004 not found")
        passed = False
    else:
        assignee = task_004.get("assigned_to", "").lower()
        if "arjun" in assignee:
            reasons.append("âœ… TASK-004 assigned to Arjun")
        else:
            reasons.append(f"âŒ TASK-004 assigned to '{assignee}', expected Arjun")
            passed = False

    sync_meetings = [
        m for m in calendar.meetings
        if m["id"] not in ("meet_001", "meet_002")
        and "arjun" in [a.lower() for a in m.get("attendees", [])]
        and "sneha" in [a.lower() for a in m.get("attendees", [])]
    ]
    if len(sync_meetings) > 0:
        reasons.append("âœ… New sync meeting booked with both engineers")
    else:
        reasons.append("âŒ No new meeting found with both Arjun and Sneha")
        passed = False

    eng_messages = chat.channels.get("engineering", [])
    agent_eng_msgs = [m for m in eng_messages if m.get("user") == "agent"]
    if len(agent_eng_msgs) > 0:
        reasons.append(f"âœ… Resolution message posted by agent in engineering channel")
    else:
        reasons.append("âŒ No resolution message posted by agent in engineering channel")
        passed = False

    return passed, reasons


def check_client_onboarding(apps: dict) -> tuple:
    """Verify the new client was actually onboarded."""
    reasons = []
    passed = True

    crm = apps["crm"]
    tasks = apps["tasks"]
    calendar = apps["calendar"]
    email = apps["email"]
    chat = apps["chat"]

    priya_contact = None
    for contact in crm.contacts.values():
        name = contact.get("name", "").lower()
        company = contact.get("company", "").lower()
        email_addr = contact.get("email", "").lower()
        if ("priya" in name
            or "newclient" in company
            or "newclient" in email_addr):
            priya_contact = contact
            break

    if priya_contact:
        reasons.append(f"âœ… Contact created for {priya_contact.get('name', 'new client')}")
    else:
        reasons.append("âŒ No contact found for Priya / NewClient Inc")
        passed = False

    seeded_ids = {"TASK-001", "TASK-002", "TASK-003", "TASK-004"}
    new_tasks = [
        t for t in tasks.tasks.values()
        if t["id"] not in seeded_ids
    ]
    if len(new_tasks) >= 1:
        titles = ", ".join(t.get("title", "")[:40] for t in new_tasks)
        reasons.append(f"âœ… New task(s) created ({len(new_tasks)}): {titles}")
    else:
        reasons.append("âŒ No new onboarding tasks were created")
        passed = False

    new_meetings = [
        m for m in calendar.meetings
        if m["id"] not in ("meet_001", "meet_002")
    ]
    if len(new_meetings) > 0:
        reasons.append(f"âœ… New meeting booked ({len(new_meetings)} meeting(s))")
    else:
        reasons.append("âŒ No new kickoff meeting booked")
        passed = False

    welcome_replies = [
        e for e in email.sent
        if "priya" in e.get("to", "").lower() or "newclient" in e.get("to", "").lower()
    ]
    if len(welcome_replies) > 0:
        reasons.append(f"âœ… Welcome reply sent to new client")
    else:
        reasons.append("âŒ No welcome reply sent to the new client")
        passed = False

    general_messages = chat.channels.get("general", [])
    agent_general_msgs = [m for m in general_messages if m.get("user") == "agent"]
    if len(agent_general_msgs) > 0:
        reasons.append("âœ… Announcement posted by agent in general channel")
    else:
        reasons.append("âŒ No announcement posted by agent in general channel")
        passed = False

    return passed, reasons


def check_morning_checkin(apps: dict) -> tuple:
    """Easy curriculum scenario â€” passes automatically if tools were called."""
    return (True, ["âœ… Morning check-in complete (no state mutations required)"])


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SCENARIOS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SCENARIOS = [
    {
        "id": "scenario_deal_rescue",
        "name": "ðŸ”¥ Deal Rescue",
        "description": "Acme Corp just emailed to cancel their $50,000 contract. "
                       "The agent must read the email, check the CRM deal, reply with a discount offer, "
                       "update the deal stage, and book a follow-up call.",
        "agent_prompt": (
            "You are an AI sales assistant at a company. "
            "A client has emailed about cancelling their contract. "
            "Your job is to:\n"
            "1. Read the inbox and find the cancellation email\n"
            "2. Look up their deal in the CRM\n"
            "3. Reply to the client with a 10% discount offer to keep them\n"
            "4. Update the deal stage to 'Negotiation' in CRM\n"
            "5. Add a note to the deal about the discount offer\n"
            "6. Book a 30-minute follow-up call with the client for 2026-04-23 at 10:00\n"
            "7. Post a message in the sales channel about this situation\n"
            "Use the available tools to complete all these steps. Be thorough. "
            "Do not repeat tool calls you have already made."
        ),
        "required_actions": [
            "read_inbox",
            "read_email",
            "get_deal",
            "reply_email",
            "update_deal_stage",
            "add_note",
            "book_meeting",
            "post_message",
        ],
        "expected_counts": {},
        "success_check": check_deal_rescue,
    },
    {
        "id": "scenario_team_conflict",
        "name": "âš”ï¸ Team Conflict",
        "description": "Two engineers â€” Arjun and Sneha â€” both claimed the same task (TASK-003). "
                       "The agent must read the conflict in chat, resolve it by reassigning tasks, "
                       "notify both engineers, and schedule a sync meeting.",
        "agent_prompt": (
            "You are an AI team manager. "
            "There is a conflict: two engineers have both claimed the same task. "
            "Your job is to:\n"
            "1. Read the engineering channel to understand the conflict\n"
            "2. Look at the conflicting task details\n"
            "3. List all tasks and find another task to give to one of the engineers\n"
            "4. Reassign TASK-003 to Sneha (she started first)\n"
            "5. Assign TASK-004 to Arjun as a replacement\n"
            "6. Post a resolution message in the engineering channel\n"
            "7. Book a 15-minute sync meeting with both engineers on 2026-04-22 at 11:00\n"
            "Use the available tools to complete all steps. "
            "Do not repeat tool calls you have already made."
        ),
        "required_actions": [
            "read_channel",
            "get_task",
            "list_tasks",
            "assign_task",
            "post_message",
            "book_meeting",
        ],
        "expected_counts": {
            "assign_task": 2,
        },
        "success_check": check_team_conflict,
    },
    {
        "id": "scenario_client_onboarding",
        "name": "ðŸš€ Client Onboarding",
        "description": "A new client (Priya Sharma from NewClient Inc) has emailed expressing interest. "
                       "The agent must read the email, create a CRM contact and deal, "
                       "create onboarding tasks, book a kickoff meeting, and send a welcome email.",
        "agent_prompt": (
            "You are an AI onboarding assistant. "
            "A new potential client has emailed your company. "
            "Your job is to:\n"
            "1. Read the inbox and find the new client email\n"
            "2. Read the full email to get their details\n"
            "3. Create a new contact in CRM for them\n"
            "4. Reply to their email with a warm welcome and next steps\n"
            "5. Create a task 'Prepare onboarding documents for NewClient Inc' assigned to arjun\n"
            "6. Create a task 'Send pricing proposal to NewClient Inc' assigned to sneha\n"
            "7. Book a 60-minute kickoff meeting with the client on 2026-04-24 at 14:00\n"
            "8. Post in general channel that a new client is being onboarded\n"
            "Use the available tools to complete all steps. "
            "Do not repeat tool calls you have already made."
        ),
        "required_actions": [
            "read_inbox",
            "read_email",
            "create_contact",
            "reply_email",
            "create_task",
            "book_meeting",
            "post_message",
        ],
        "expected_counts": {
            "create_task": 2,
        },
        "success_check": check_client_onboarding,
    },
    {
        "id": "scenario_morning_checkin",
        "name": "â˜€ï¸ Morning Check-in",
        "description": "Start of the workday. The agent reads the inbox and lists "
                       "chat channels to get oriented. Curriculum scenario for "
                       "easy early reward during training.",
        "agent_prompt": (
            "You are starting your workday as an AI employee. "
            "To get oriented, do exactly two things:\n"
            "1. Read your inbox to see what emails arrived\n"
            "2. List the available team chat channels\n"
            "After those two tool calls, you're done. Do not call any other tools."
        ),
        "required_actions": [
            "read_inbox",
            "list_channels",
        ],
        "expected_counts": {},
        "success_check": check_morning_checkin,
    },
]
