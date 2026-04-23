\# Reward Hacking Audit



Per build-guide Section 8 ("Protect yourself against reward hacking"),

we analyzed each of our 4 reward functions for exploitation risk.



\## Summary



Our reward stack is mostly robust because each function targets a

different axis of agent behavior:



| Reward Function | Range | What it catches |

|-----------------|-------|-----------------|

| tool\_correctness | 0 to N | Was each required tool called? |

| tool\_efficiency  | -K to 0 | Unnecessary or duplicate calls? |

| task\_completion  | 0 or +10 | Did state match the goal? |

| format\_validity  | 0 or +2 | Were tool args well-formed? |



An agent attempting to hack any single axis loses points on others.



\## Protections in place (from build guide Section 8)



\- ✅ Multiple independent reward functions (4)

\- ✅ Step cap: MAX\_EPISODE\_STEPS = 40 (safety net against loops)

\- ✅ Execution locked: agent can only call defined tools through env.step()

\- ✅ No global state abuse: each episode creates fresh app instances

\- ✅ Unknown tools rejected and NOT added to trajectory (can't pad rewards)



\## Known limitation



Our state-based success checks verify structural outcomes (did a reply

get sent? did deal\_stage change?) but NOT content quality. An agent

could pass checks with low-effort content like "abc" as an email body.



\*\*Future work:\*\* Add LLM-as-judge (per build-guide Section 9) to score

content quality as a 5th reward function. Judge scores are themselves

gameable, so they should supplement — not replace — the existing

structural checks.

