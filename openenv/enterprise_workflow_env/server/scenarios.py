# ============================================================
# scenarios.py — The 3 Task Scenarios
# Each scenario is a dict that tells the agent what to do,
# defines what "success" looks like as required tool calls,
# AND includes a state-based check that inspects the final
# app state to verify real outcomes (not just tool calls).
# ============================================================


# ─────────────────────────────────────────────
# STATE-BASED SUCCESS CHECKS
# Each returns (passed: bool, reasons: list[str])
# reasons lists what was checked and whether it passed/failed
# ─────────────────────────────────────────────

def check_deal_rescue(apps: dict) -> tuple:
    """Verify the Acme deal was actually rescued — not just that tools were called."""
    reasons = []
    passed = True

    crm = apps["crm"]
    calendar = apps["calendar"]
    email = apps["email"]
    chat = apps["chat"]

    # 1. Find the Acme deal (deal_001 in seed data)
    acme_deal = None
    for deal in crm.deals.values():
        if "acme" in deal.get("company", "").lower():
            acme_deal = deal
            break

    if acme_deal is None:
        reasons.append("❌ No Acme deal found in CRM")
        passed = False
    else:
        # 1a. Deal stage should be updated away from 'Active'
        stage = acme_deal.get("stage", "")
        if stage in ("Negotiation", "At Risk"):
            reasons.append(f"✅ Acme deal stage updated to '{stage}'")
        else:
            reasons.append(f"❌ Acme deal stage is still '{stage}', expected Negotiation or At Risk")
            passed = False

        # 1b. At least one NEW note should have been added (seed has 2 notes)
        notes = acme_deal.get("notes", [])
        if len(notes) > 2:
            reasons.append(f"✅ New note added to Acme deal ({len(notes)} notes total)")
        else:
            reasons.append(f"❌ No new note added to Acme deal (still {len(notes)} notes)")
            passed = False

    # 2. A follow-up meeting should exist (seed calendar has 2 meetings)
    follow_up_meetings = [
        m for m in calendar.meetings
        if m["id"] not in ("meet_001", "meet_002")
    ]
    if len(follow_up_meetings) > 0:
        reasons.append(f"✅ Follow-up meeting booked ({len(follow_up_meetings)} new meeting(s))")
    else:
        reasons.append("❌ No new follow-up meeting was booked")
        passed = False

    # 3. A reply to the Acme cancellation email should have been sent
    acme_replies = [
        e for e in email.sent
        if "acme" in e.get("to", "").lower() or "rajesh" in e.get("to", "").lower()
    ]
    if len(acme_replies) > 0:
        reasons.append(f"✅ Reply email sent to Acme ({len(acme_replies)} email(s))")
    else:
        reasons.append("❌ No reply email was sent to the Acme client")
        passed = False

    # 4. A message should be posted in the sales channel (seed has 1 message)
    sales_messages = chat.channels.get("sales", [])
    agent_sales_msgs = [m for m in sales_messages if m.get("user") == "agent"]
    if len(agent_sales_msgs) > 0:
        reasons.append(f"✅ Agent posted in sales channel ({len(agent_sales_msgs)} message(s))")
    else:
        reasons.append("❌ No message posted by agent in sales channel")
        passed = False

    return passed, reasons


def check_team_conflict(apps: dict) -> tuple:
    """Verify the task conflict was actually resolved."""
    reasons = []
    passed = True

    tasks = apps["tasks"]
    calendar = apps["calendar"]
    chat = apps["chat"]

    # 1. TASK-003 should be reassigned to sneha (was arjun in seed)
    task_003 = tasks.tasks.get("TASK-003")
    if task_003 is None:
        reasons.append("❌ TASK-003 not found")
        passed = False
    else:
        assignee = task_003.get("assigned_to", "").lower()
        if "sneha" in assignee:
            reasons.append("✅ TASK-003 reassigned to Sneha")
        else:
            reasons.append(f"❌ TASK-003 assigned to '{assignee}', expected Sneha")
            passed = False

    # 2. TASK-004 should be reassigned to arjun (was 'unassigned' in seed)
    task_004 = tasks.tasks.get("TASK-004")
    if task_004 is None:
        reasons.append("❌ TASK-004 not found")
        passed = False
    else:
        assignee = task_004.get("assigned_to", "").lower()
        if "arjun" in assignee:
            reasons.append("✅ TASK-004 assigned to Arjun")
        else:
            reasons.append(f"❌ TASK-004 assigned to '{assignee}', expected Arjun")
            passed = False

    # 3. A sync meeting should be booked with both arjun and sneha (new meeting, not seeded)
    sync_meetings = [
        m for m in calendar.meetings
        if m["id"] not in ("meet_001", "meet_002")
        and "arjun" in [a.lower() for a in m.get("attendees", [])]
        and "sneha" in [a.lower() for a in m.get("attendees", [])]
    ]
    if len(sync_meetings) > 0:
        reasons.append("✅ New sync meeting booked with both engineers")
    else:
        reasons.append("❌ No new meeting found with both Arjun and Sneha")
        passed = False

    # 4. A resolution message should be posted in the engineering channel by the agent
    eng_messages = chat.channels.get("engineering", [])
    agent_eng_msgs = [m for m in eng_messages if m.get("user") == "agent"]
    if len(agent_eng_msgs) > 0:
        reasons.append(f"✅ Resolution message posted by agent in engineering channel")
    else:
        reasons.append("❌ No resolution message posted by agent in engineering channel")
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

    # 1. A contact for Priya / NewClient Inc should exist in CRM (seed only has rajesh)
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
        reasons.append(f"✅ Contact created for {priya_contact.get('name', 'new client')}")
    else:
        reasons.append("❌ No contact found for Priya / NewClient Inc")
        passed = False

    # 2. At least one new onboarding-related task should exist (seed has TASK-001..004)
    seeded_ids = {"TASK-001", "TASK-002", "TASK-003", "TASK-004"}
    new_tasks = [
        t for t in tasks.tasks.values()
        if t["id"] not in seeded_ids
    ]
    if len(new_tasks) >= 1:
        titles = ", ".join(t.get("title", "")[:40] for t in new_tasks)
        reasons.append(f"✅ New task(s) created ({len(new_tasks)}): {titles}")
    else:
        reasons.append("❌ No new onboarding tasks were created")
        passed = False

    # 3. A kickoff meeting should be booked (new meeting, not seeded)
    new_meetings = [
        m for m in calendar.meetings
        if m["id"] not in ("meet_001", "meet_002")
    ]
    if len(new_meetings) > 0:
        reasons.append(f"✅ New meeting booked ({len(new_meetings)} meeting(s))")
    else:
        reasons.append("❌ No new kickoff meeting booked")
        passed = False

    # 4. A welcome reply should have been sent to priya / newclient
    welcome_replies = [
        e for e in email.sent
        if "priya" in e.get("to", "").lower() or "newclient" in e.get("to", "").lower()
    ]
    if len(welcome_replies) > 0:
        reasons.append(f"✅ Welcome reply sent to new client")
    else:
        reasons.append("❌ No welcome reply sent to the new client")
        passed = False

    # 5. A message should be posted in the general channel by the agent
    general_messages = chat.channels.get("general", [])
    agent_general_msgs = [m for m in general_messages if m.get("user") == "agent"]
    if len(agent_general_msgs) > 0:
        reasons.append("✅ Announcement posted by agent in general channel")
    else:
        reasons.append("❌ No announcement posted by agent in general channel")
        passed = False

    return passed, reasons


# ─────────────────────────────────────────────
# SCENARIOS
#
# expected_counts: optional dict that tells the reward function how many
# times each tool is legitimately expected to be called. If a tool is
# called more than this count, the extras are penalized. If a tool isn't
# listed here, it defaults to 1 (i.e. any repeat is an extra).
# This lets scenarios that genuinely need multiple calls to the same
# tool (e.g. creating two different tasks) score fully.
# ─────────────────────────────────────────────

SCENARIOS = [
    {
        "id": "scenario_deal_rescue",
        "name": "🔥 Deal Rescue",
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
        "expected_counts": {
            # Deal Rescue: each tool is called exactly once.
            # No override needed — all defaults are 1.
        },
        "success_check": check_deal_rescue,
    },
    {
        "id": "scenario_team_conflict",
        "name": "⚔️ Team Conflict",
        "description": "Two engineers — Arjun and Sneha — both claimed the same task (TASK-003). "
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
            # Two reassignments: TASK-003 → Sneha, TASK-004 → Arjun
            "assign_task": 2,
        },
        "success_check": check_team_conflict,
    },
    {
        "id": "scenario_client_onboarding",
        "name": "🚀 Client Onboarding",
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
            # Two tasks to create: one for arjun, one for sneha
            "create_task": 2,
        },
        "success_check": check_client_onboarding,
    },
]