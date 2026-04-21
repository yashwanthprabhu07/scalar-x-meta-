# ============================================================
# scenarios.py — The 3 Task Scenarios
# Each scenario is a dict that tells the agent what to do
# and defines what "success" looks like.
# ============================================================

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
            "Use the available tools to complete all these steps. Be thorough."
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
            "Use the available tools to complete all steps."
        ),
        "required_actions": [
            "read_channel",
            "get_task",
            "list_tasks",
            "assign_task",
            "post_message",
            "book_meeting",
        ],
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
            "Use the available tools to complete all steps."
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
    },
]
