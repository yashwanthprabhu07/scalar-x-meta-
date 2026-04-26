# ============================================================
# client.py — HTTP/WebSocket client for the Enterprise Workflow Env
#
# This is what external callers (training scripts, test harnesses,
# dashboards) use to talk to the environment once it's running.
#
# Usage:
#   from enterprise_workflow_env.client import EnterpriseWorkflowEnv
#   from enterprise_workflow_env.models  import EnterpriseAction
#
#   with EnterpriseWorkflowEnv(base_url="http://localhost:8000") as env:
#       result = env.reset()
#       print("scenario:", result.observation.scenario_id)
#
#       result = env.step(EnterpriseAction(tool_name="read_inbox"))
#       result = env.step(EnterpriseAction(tool_name="read_email",
#                                          tool_args={"email_id": "email_001"}))
#       # ... etc
#
# Or connect to a Docker image directly:
#   env = EnterpriseWorkflowEnv.from_docker_image(
#       "enterprise_workflow_env-env:latest"
#   )
# ============================================================

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EnterpriseAction, EnterpriseObservation


class EnterpriseWorkflowEnv(
    EnvClient[EnterpriseAction, EnterpriseObservation, State]
):
    """
    Client for the Enterprise Workflow Environment.

    Opens a persistent WebSocket connection to the server. Each client
    instance gets its own dedicated environment session, so you can
    run multiple clients against the same server for parallel training
    rollouts (which is exactly what TRL's GRPOTrainer does).
    """

    def _step_payload(self, action: EnterpriseAction) -> Dict[str, Any]:
        """Convert an EnterpriseAction into a JSON-serializable dict."""
        return {
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EnterpriseObservation]:
        """Build a StepResult from the server's JSON response."""
        obs_data = payload.get("observation", {}) or {}

        observation = EnterpriseObservation(
            # Our custom fields
            tool_result=obs_data.get("tool_result", ""),
            error=obs_data.get("error", ""),
            scenario_id=obs_data.get("scenario_id", ""),
            taken_actions=obs_data.get("taken_actions", []),
            step_count=obs_data.get("step_count", 0),
            breakdown=obs_data.get("breakdown", []),
            task_success=obs_data.get("task_success"),
            # OpenEnv standard fields
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Build a State from the server's /state response."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )