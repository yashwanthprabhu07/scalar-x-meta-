# ============================================================
# server/app.py — FastAPI application for the Enterprise Workflow Env
#
# Exposes the EnterpriseWorkflowEnvironment over HTTP and WebSocket
# via OpenEnv's create_app helper.
#
# Endpoints (handled by OpenEnv's http_server):
#   POST  /reset     — start a new episode
#   POST  /step      — execute one action (tool call)
#   GET   /state     — get episode_id + step_count
#   GET   /schema    — action/observation JSON schemas
#   GET   /health    — health check
#   WS    /ws        — persistent session (used by EnvClient)
#
# Usage:
#   cd openenv/enterprise_workflow_env
#   uvicorn server.app:app --host 0.0.0.0 --port 8000
# ============================================================

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

# Import our custom models and environment.
# Try package-relative imports first (for when the env is imported as a
# module by tooling), then fall back to importing from the current
# package/sys.path (how uvicorn loads it when run as `server.app:app`).
try:
    from ..models import EnterpriseAction, EnterpriseObservation
    from .enterprise_workflow_env_environment import EnterpriseWorkflowEnvironment
except ImportError:
    # uvicorn loads this as `server.app`, which means `server` is the
    # top-level package and `..models` looks "above" it — not allowed.
    # In that case, ensure the parent dir is on sys.path and import
    # models directly.
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from models import EnterpriseAction, EnterpriseObservation  # type: ignore
    from server.enterprise_workflow_env_environment import EnterpriseWorkflowEnvironment  # type: ignore


# Create the FastAPI app with web interface enabled.
# max_concurrent_envs controls how many simultaneous WebSocket sessions
# the server supports — we set it higher than the default so TRL GRPO
# training (which opens N connections for N generations) works.
app = create_app(
    EnterpriseWorkflowEnvironment,
    EnterpriseAction,
    EnterpriseObservation,
    env_name="enterprise_workflow_env",
    max_concurrent_envs=8,
)


def main():
    """Direct-run entrypoint (python -m server.app)."""
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()