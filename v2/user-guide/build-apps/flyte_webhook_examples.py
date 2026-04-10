# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0",
#    "fastapi",
#    "uvicorn",
#    "httpx",
# ]
# ///

"""Examples showing how to use FlyteWebhookAppEnvironment."""

import logging

import flyte
import flyte.app
from flyte.app.extras import FlyteWebhookAppEnvironment

# {{docs-fragment basic-webhook}}
webhook_env = FlyteWebhookAppEnvironment(
    name="my-webhook",
    title="My Flyte Webhook",
    description="A pre-built webhook service for Flyte operations",
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
    scaling=flyte.app.Scaling(replicas=1),
)
# {{/docs-fragment basic-webhook}}

# {{docs-fragment endpoint-groups}}
task_runner_webhook = FlyteWebhookAppEnvironment(
    name="task-runner-webhook",
    title="Task Runner Webhook",
    endpoint_groups=["core", "task", "run"],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
)
# {{/docs-fragment endpoint-groups}}

# {{docs-fragment individual-endpoints}}
minimal_webhook = FlyteWebhookAppEnvironment(
    name="minimal-webhook",
    title="Minimal Webhook",
    endpoints=["health", "run_task", "get_run"],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
)
# {{/docs-fragment individual-endpoints}}

# {{docs-fragment task-allowlist}}
restricted_webhook = FlyteWebhookAppEnvironment(
    name="restricted-webhook",
    title="Restricted Webhook",
    endpoint_groups=["core", "task", "run"],
    task_allowlist=[
        "production/my-project/allowed-task",
        "my-project/another-task",
        "any-domain-task",
    ],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
)
# {{/docs-fragment task-allowlist}}

# {{docs-fragment app-allowlist}}
app_manager_webhook = FlyteWebhookAppEnvironment(
    name="app-manager-webhook",
    title="App Manager Webhook",
    endpoint_groups=["core", "app"],
    app_allowlist=["my-app", "another-app"],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
)
# {{/docs-fragment app-allowlist}}

# {{docs-fragment trigger-allowlist}}
trigger_manager_webhook = FlyteWebhookAppEnvironment(
    name="trigger-manager-webhook",
    title="Trigger Manager Webhook",
    endpoint_groups=["core", "trigger"],
    trigger_allowlist=["my-task/my-trigger", "another-trigger"],
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=True,
)
# {{/docs-fragment trigger-allowlist}}

# {{docs-fragment deploy-webhook}}
if __name__ == "__main__":
    import os

    import httpx

    flyte.init_from_config(log_level=logging.DEBUG)

    served_app = flyte.serve(webhook_env)
    url = served_app.url
    endpoint = served_app.endpoint
    print(f"Webhook is served on {url}")
    print(f"OpenAPI docs available at: {endpoint}/docs")

    served_app.activate(wait=True)
# {{/docs-fragment deploy-webhook}}

# {{docs-fragment call-webhook}}
    token = os.getenv("FLYTE_API_KEY")
    if not token:
        raise ValueError("FLYTE_API_KEY not set. Obtain with: flyte get api-key")

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "flyte-webhook-client/1.0",
    }

    with httpx.Client(headers=headers) as client:
        # Health check (no auth required)
        health = client.get(f"{endpoint}/health")
        print(f"/health: {health.json()}")

        # Get current user info (requires auth)
        me = client.get(f"{endpoint}/me")
        print(f"/me: {me.json()}")

        # Run a task
        resp = client.post(
            f"{endpoint}/run-task/development/my-project/my-task",
            json={"x": 42, "y": "hello"},
        )
        result = resp.json()
        print(f"Run task: {result}")

        # Check run status
        run_name = result["name"]
        run = client.get(f"{endpoint}/run/{run_name}")
        print(f"Run status: {run.json()}")
# {{/docs-fragment call-webhook}}
