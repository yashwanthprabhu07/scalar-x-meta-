# ============================================================
# scenarios.py — The Task Scenarios (with randomization)
# ============================================================

import random

_CLIENTS = [
    {"name": "Rajesh Kumar",    "company": "Acme Corp",          "email": "rajesh.kumar@acmecorp.com"},
    {"name": "Meera Singh",     "company": "TechWave Ltd",        "email": "meera.singh@techwave.com"},
    {"name": "Aryan Patel",     "company": "GlobalSoft Inc",      "email": "aryan.patel@globalsoft.com"},
    {"name": "Divya Nair",      "company": "Nexus Systems",       "email": "divya.nair@nexussystems.com"},
    {"name": "Kiran Sharma",    "company": "BlueChip Corp",       "email": "kiran.sharma@bluechip.com"},
    {"name": "Sunita Reddy",    "company": "Pinnacle Tech",       "email": "sunita.reddy@pinnacletech.com"},
    {"name": "Vikram Malhotra", "company": "Horizon Ventures",    "email": "vikram.malhotra@horizonv.com"},
    {"name": "Ananya Krishnan", "company": "Stellar Solutions",   "email": "ananya.k@stellarsol.com"},
    {"name": "Rohit Bansal",    "company": "CoreEdge Systems",    "email": "rohit.bansal@coreedge.com"},
    {"name": "Pooja Iyer",      "company": "Vanguard Analytics",  "email": "pooja.iyer@vanguarda.com"},
    {"name": "Sanjay Menon",    "company": "Titan Enterprises",   "email": "sanjay.menon@titanent.com"},
    {"name": "Kavitha Rao",     "company": "Apex Digital",        "email": "kavitha.rao@apexdigital.com"},
    {"name": "Aditya Joshi",    "company": "Synergy Corp",        "email": "aditya.joshi@synergycorp.com"},
    {"name": "Nisha Verma",     "company": "Quantum Leap Ltd",    "email": "nisha.verma@quantumleap.com"},
    {"name": "Ravi Chandran",   "company": "Momentum Group",      "email": "ravi.chandran@momentumg.com"},
    {"name": "Deepa Subramaniam","company": "Everest Technologies","email": "deepa.s@everesttech.com"},
    {"name": "Manoj Gupta",     "company": "Crest Innovations",   "email": "manoj.gupta@crestinnov.com"},
    {"name": "Shruti Kapoor",   "company": "Zenith Solutions",    "email": "shruti.kapoor@zenithsol.com"},
    {"name": "Rahul Desai",     "company": "Catalyst Systems",    "email": "rahul.desai@catalystsys.com"},
    {"name": "Priya Nambiar",   "company": "Luminary Corp",       "email": "priya.nambiar@luminarycorp.com"},
]

_NEW_CLIENTS = [
    {"name": "Priya Sharma",    "company": "NewClient Inc",    "email": "priya.sharma@newclient.com",    "interest": "enterprise analytics platform"},
    {"name": "Rohan Gupta",     "company": "StartupXYZ",       "email": "rohan.gupta@startupxyz.com",    "interest": "team collaboration tools"},
    {"name": "Anita Desai",     "company": "Vertex Partners",  "email": "anita.desai@vertexpartners.com", "interest": "CRM and sales automation"},
    {"name": "Suresh Iyer",     "company": "CloudFirst Ltd",   "email": "suresh.iyer@cloudfirst.com",    "interest": "cloud infrastructure management"},
    {"name": "Lakshmi Rao",     "company": "DataDriven Co",    "email": "lakshmi.rao@datadriven.com",    "interest": "data analytics and BI"},
    {"name": "Nikhil Mehta",    "company": "BrightPath AI",    "email": "nikhil.mehta@brightpathai.com", "interest": "AI-powered workflow automation"},
    {"name": "Swathi Pillai",   "company": "GreenLeaf Tech",   "email": "swathi.pillai@greenleaf.com",   "interest": "sustainable supply chain software"},
    {"name": "Arjun Nair",      "company": "PeakFlow Systems", "email": "arjun.nair@peakflow.com",       "interest": "real-time operational dashboards"},
    {"name": "Tanya Bhatt",     "company": "Skyline Ventures", "email": "tanya.bhatt@skylinev.com",      "interest": "HR and talent management tools"},
    {"name": "Vivek Shetty",    "company": "RapidScale Inc",   "email": "vivek.shetty@rapidscale.com",   "interest": "DevOps and CI/CD automation"},
    {"name": "Meghna Choudhary","company": "Opal Analytics",   "email": "meghna.c@opalanalytics.com",   "interest": "customer retention analytics"},
    {"name": "Pratik Shah",     "company": "Forge Digital",    "email": "pratik.shah@forgedigital.com",  "interest": "e-commerce platform integration"},
    {"name": "Kavya Menon",     "company": "Cascade Solutions","email": "kavya.menon@cascadesol.com",    "interest": "enterprise resource planning"},
    {"name": "Arun Krishnamurthy","company": "Elevate Corp",   "email": "arun.k@elevatecorp.com",        "interest": "financial forecasting tools"},
    {"name": "Shalini Pandey",  "company": "Orbit Innovations","email": "shalini.pandey@orbitinnov.com", "interest": "IoT device management platform"},
]

_DEALS = [30000, 40000, 45000, 50000, 60000, 75000, 80000]
_DISCOUNTS = [5, 10, 15]
_MEETING_DATES = ["2026-04-23", "2026-04-24", "2026-04-25", "2026-04-28", "2026-04-29"]
_MEETING_TIMES = ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]
_CONFLICT_REASONS = [
    "Both engineers picked up the same task from the backlog simultaneously.",
    "A miscommunication in standup led to duplicate task assignments.",
    "The task board was not updated, causing both engineers to start work.",
]


# ─────────────────────────────────────────────
# STATE-BASED SUCCESS CHECKS
# ─────────────────────────────────────────────

def check_deal_rescue(apps, client):
    reasons = []
    passed = True
    crm = apps["crm"]
    calendar = apps["calendar"]
    email = apps["email"]
    chat = apps["chat"]

    client_deal = None
    for deal in crm.deals.values():
        if client["company"].lower() in deal.get("company", "").lower():
            client_deal = deal
            break
    if client_deal is None:
        for deal in crm.deals.values():
            client_deal = deal
            break

    if client_deal is None:
        reasons.append("No deal found in CRM")
        passed = False
    else:
        stage = client_deal.get("stage", "")
        if stage in ("Negotiation", "At Risk"):
            reasons.append("Deal stage updated to " + stage)
        else:
            reasons.append("Deal stage is still " + stage + ", expected Negotiation or At Risk")
            passed = False
        notes = client_deal.get("notes", [])
        if len(notes) > 2:
            reasons.append("New note added to deal (" + str(len(notes)) + " notes total)")
        else:
            reasons.append("No new note added (still " + str(len(notes)) + " notes)")
            passed = False

    follow_up = [m for m in calendar.meetings if m["id"] not in ("meet_001", "meet_002")]
    if follow_up:
        reasons.append("Follow-up meeting booked (" + str(len(follow_up)) + " new meeting(s))")
    else:
        reasons.append("No new follow-up meeting was booked")
        passed = False

    client_replies = [
        e for e in email.sent
        if client["email"].lower() in e.get("to", "").lower()
        or client["name"].split()[0].lower() in e.get("to", "").lower()
    ]
    if client_replies:
        reasons.append("Reply email sent to client (" + str(len(client_replies)) + " email(s))")
    else:
        reasons.append("No reply email sent to the client")
        passed = False

    sales_msgs = [m for m in chat.channels.get("sales", []) if m.get("user") == "agent"]
    if sales_msgs:
        reasons.append("Agent posted in sales channel (" + str(len(sales_msgs)) + " message(s))")
    else:
        reasons.append("No message posted by agent in sales channel")
        passed = False

    return passed, reasons


def check_team_conflict(apps):
    reasons = []
    passed = True
    tasks = apps["tasks"]
    calendar = apps["calendar"]
    chat = apps["chat"]

    t003 = tasks.tasks.get("TASK-003")
    if t003 is None:
        reasons.append("TASK-003 not found")
        passed = False
    else:
        assignee = t003.get("assigned_to", "").lower()
        if "sneha" in assignee:
            reasons.append("TASK-003 reassigned to Sneha")
        else:
            reasons.append("TASK-003 assigned to " + assignee + ", expected Sneha")
            passed = False

    t004 = tasks.tasks.get("TASK-004")
    if t004 is None:
        reasons.append("TASK-004 not found")
        passed = False
    else:
        assignee = t004.get("assigned_to", "").lower()
        if "arjun" in assignee:
            reasons.append("TASK-004 assigned to Arjun")
        else:
            reasons.append("TASK-004 assigned to " + assignee + ", expected Arjun")
            passed = False

    sync = [
        m for m in calendar.meetings
        if m["id"] not in ("meet_001", "meet_002")
        and "arjun" in [a.lower() for a in m.get("attendees", [])]
        and "sneha" in [a.lower() for a in m.get("attendees", [])]
    ]
    if sync:
        reasons.append("New sync meeting booked with both engineers")
    else:
        reasons.append("No new meeting found with both Arjun and Sneha")
        passed = False

    eng_msgs = [m for m in chat.channels.get("engineering", []) if m.get("user") == "agent"]
    if eng_msgs:
        reasons.append("Resolution message posted in engineering channel")
    else:
        reasons.append("No resolution message posted in engineering channel")
        passed = False

    return passed, reasons


def check_client_onboarding(apps, client):
    reasons = []
    passed = True
    crm = apps["crm"]
    tasks = apps["tasks"]
    calendar = apps["calendar"]
    email = apps["email"]
    chat = apps["chat"]

    first = client["name"].split()[0].lower()
    co = client["company"].split()[0].lower()

    new_contact = None
    for contact in crm.contacts.values():
        n = contact.get("name", "").lower()
        c = contact.get("company", "").lower()
        e = contact.get("email", "").lower()
        if first in n or co in c or co in e:
            new_contact = contact
            break

    if new_contact:
        reasons.append("Contact created for " + new_contact.get("name", "new client"))
    else:
        reasons.append("No contact found for " + client["name"] + " / " + client["company"])
        passed = False

    seeded = {"TASK-001", "TASK-002", "TASK-003", "TASK-004"}
    new_tasks = [t for t in tasks.tasks.values() if t["id"] not in seeded]
    if len(new_tasks) >= 1:
        titles = ", ".join(t.get("title", "")[:40] for t in new_tasks)
        reasons.append("New task(s) created (" + str(len(new_tasks)) + "): " + titles)
    else:
        reasons.append("No new onboarding tasks created")
        passed = False

    new_meetings = [m for m in calendar.meetings if m["id"] not in ("meet_001", "meet_002")]
    if new_meetings:
        reasons.append("New meeting booked (" + str(len(new_meetings)) + " meeting(s))")
    else:
        reasons.append("No new kickoff meeting booked")
        passed = False

    welcome = [
        e for e in email.sent
        if first in e.get("to", "").lower() or co in e.get("to", "").lower()
    ]
    if welcome:
        reasons.append("Welcome reply sent to new client")
    else:
        reasons.append("No welcome reply sent to the new client")
        passed = False

    gen_msgs = [m for m in chat.channels.get("general", []) if m.get("user") == "agent"]
    if gen_msgs:
        reasons.append("Announcement posted in general channel")
    else:
        reasons.append("No announcement posted in general channel")
        passed = False

    return passed, reasons


def check_morning_checkin(apps):
    return (True, ["Morning check-in complete (no state mutations required)"])


# ─────────────────────────────────────────────
# SCENARIO GENERATORS
# ─────────────────────────────────────────────

def generate_deal_rescue():
    client = random.choice(_CLIENTS)
    amount = random.choice(_DEALS)
    discount = random.choice(_DISCOUNTS)
    meet_date = random.choice(_MEETING_DATES)
    meet_time = random.choice(_MEETING_TIMES)
    return {
        "id": "scenario_deal_rescue",
        "name": "Deal Rescue",
        "description": (
            client["company"] + " just emailed to cancel their $" + str(amount) + " contract. "
            "The agent must read the email, check the CRM deal, reply with a " + str(discount) +
            "% discount offer, update the deal stage, and book a follow-up call."
        ),
        "agent_prompt": (
            "You are an AI sales assistant at a company. "
            "A client (" + client["name"] + " from " + client["company"] + ") has emailed about cancelling their contract.\n"
            "Your job is to:\n"
            "1. Read the inbox and find the cancellation email\n"
            "2. Look up their deal in the CRM\n"
            "3. Reply to the client with a " + str(discount) + "% discount offer to keep them\n"
            "4. Update the deal stage to Negotiation in CRM\n"
            "5. Add a note to the deal about the discount offer\n"
            "6. Book a 30-minute follow-up call with the client for " + meet_date + " at " + meet_time + "\n"
            "7. Post a message in the sales channel about this situation\n"
            "Use the available tools to complete all these steps. Be thorough. "
            "Do not repeat tool calls you have already made."
        ),
        "required_actions": [
            "read_inbox", "read_email", "get_deal", "reply_email",
            "update_deal_stage", "add_note", "book_meeting", "post_message",
        ],
        "expected_counts": {},
        "success_check": lambda apps, c=client: check_deal_rescue(apps, c),
        "_meta": {"client": client, "amount": amount, "discount": discount},
    }


def generate_team_conflict():
    meet_date = random.choice(_MEETING_DATES)
    meet_time = random.choice(["10:00", "11:00", "14:00", "15:00"])
    reason = random.choice(_CONFLICT_REASONS)
    return {
        "id": "scenario_team_conflict",
        "name": "Team Conflict",
        "description": (
            "Two engineers, Arjun and Sneha, both claimed the same task (TASK-003). "
            + reason + " The agent must resolve the conflict by reassigning tasks and scheduling a sync."
        ),
        "agent_prompt": (
            "You are an AI team manager. "
            "There is a conflict: two engineers have both claimed the same task.\n"
            "Background: " + reason + "\n"
            "Your job is to:\n"
            "1. Read the engineering channel to understand the conflict\n"
            "2. Look at the conflicting task details\n"
            "3. List all tasks and find another task to give to one of the engineers\n"
            "4. Reassign TASK-003 to Sneha (she started first)\n"
            "5. Assign TASK-004 to Arjun as a replacement\n"
            "6. Post a resolution message in the engineering channel\n"
            "7. Book a 15-minute sync meeting with both engineers on " + meet_date + " at " + meet_time + "\n"
            "Use the available tools to complete all steps. "
            "Do not repeat tool calls you have already made."
        ),
        "required_actions": [
            "read_channel", "get_task", "list_tasks",
            "assign_task", "post_message", "book_meeting",
        ],
        "expected_counts": {"assign_task": 2},
        "success_check": lambda apps: check_team_conflict(apps),
    }


def generate_client_onboarding():
    client = random.choice(_NEW_CLIENTS)
    meet_date = random.choice(_MEETING_DATES)
    meet_time = random.choice(["13:00", "14:00", "15:00", "16:00"])
    return {
        "id": "scenario_client_onboarding",
        "name": "Client Onboarding",
        "description": (
            "A new client (" + client["name"] + " from " + client["company"] + ") has emailed expressing "
            "interest in " + client["interest"] + ". The agent must read the email, create a CRM contact, "
            "create onboarding tasks, book a kickoff meeting, and send a welcome email."
        ),
        "agent_prompt": (
            "You are an AI onboarding assistant. "
            "A new potential client has emailed your company.\n"
            "Client: " + client["name"] + " from " + client["company"] + " (" + client["email"] + ")\n"
            "They are interested in: " + client["interest"] + "\n"
            "Your job is to:\n"
            "1. Read the inbox and find the new client email\n"
            "2. Read the full email to get their details\n"
            "3. Create a new contact in CRM for them\n"
            "4. Reply to their email with a warm welcome and next steps\n"
            "5. Create a task for preparing onboarding documents assigned to arjun\n"
            "6. Create a task for sending pricing proposal assigned to sneha\n"
            "7. Book a 60-minute kickoff meeting with the client on " + meet_date + " at " + meet_time + "\n"
            "8. Post in general channel that a new client is being onboarded\n"
            "Use the available tools to complete all steps. "
            "Do not repeat tool calls you have already made."
        ),
        "required_actions": [
            "read_inbox", "read_email", "create_contact", "reply_email",
            "create_task", "book_meeting", "post_message",
        ],
        "expected_counts": {"create_task": 2},
        "success_check": lambda apps, c=client: check_client_onboarding(apps, c),
        "_meta": {"client": client},
    }


def generate_morning_checkin():
    return {
        "id": "scenario_morning_checkin",
        "name": "Morning Check-in",
        "description": "Start of the workday. Read inbox and list chat channels to get oriented.",
        "agent_prompt": (
            "You are starting your workday as an AI employee. "
            "To get oriented, do exactly two things:\n"
            "1. Read your inbox to see what emails arrived\n"
            "2. List the available team chat channels\n"
            "After those two tool calls, you are done. Do not call any other tools."
        ),
        "required_actions": ["read_inbox", "list_channels"],
        "expected_counts": {},
        "success_check": lambda apps: check_morning_checkin(apps),
    }


# ─────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────

_GENERATORS = {
    "scenario_deal_rescue":       generate_deal_rescue,
    "scenario_team_conflict":     generate_team_conflict,
    "scenario_client_onboarding": generate_client_onboarding,
    "scenario_morning_checkin":   generate_morning_checkin,
}

SCENARIOS = [fn() for fn in _GENERATORS.values()]


def get_scenario_instance(scenario_id):
    generator = _GENERATORS.get(scenario_id, generate_deal_rescue)
    return generator()


def list_scenario_ids():
    return list(_GENERATORS.keys())
