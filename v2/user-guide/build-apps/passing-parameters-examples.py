# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "fastapi",
#    "scikit-learn",
#    "joblib",
# ]
# ///

"""Examples showing different ways to pass parameters into apps."""

import os

import flyte
import flyte.app
import flyte.io
from flyte.app import Parameter, get_parameter

# {{docs-fragment basic-parameter-types}}
# String parameters
app_env = flyte.app.AppEnvironment(
    name="configurable-app",
    parameters=[
        Parameter(name="environment", value="production"),
        Parameter(name="log_level", value="INFO"),
    ],
    # ...
)

# File parameters
app_env2 = flyte.app.AppEnvironment(
    name="app-with-model",
    parameters=[
        Parameter(
            name="model_file",
            value=flyte.io.File.from_existing_remote("s3://bucket/models/model.pkl"),
            mount="/app/models/",
        ),
    ],
    # ...
)

# Directory parameters
app_env3 = flyte.app.AppEnvironment(
    name="app-with-data",
    parameters=[
        Parameter(
            name="data_dir",
            value=flyte.io.Dir.from_existing_remote("s3://bucket/data/"),
            mount="/app/data",
        ),
    ],
    # ...
)
# {{/docs-fragment basic-parameter-types}}

# {{docs-fragment parameter-access-methods}}
from fastapi import FastAPI
from flyte.app.extras import FastAPIAppEnvironment

DATA_MOUNT_PATH = "/tmp/data_file.txt"
DATA_ENV_VAR = "DATA_FILE_PATH"

demo_app = FastAPI()

demo_env = FastAPIAppEnvironment(
    name="parameter-access-demo",
    app=demo_app,
    parameters=[
        # With mount and env_var: the file is downloaded to the mount path,
        # and an environment variable is set to point to the file location.
        Parameter(
            name="data",
            type="file",
            mount=DATA_MOUNT_PATH,
            env_var=DATA_ENV_VAR,
        ),
        # With no mount or env_var: accessible only via get_parameter,
        # which returns the path to the downloaded file.
        Parameter(name="data_raw", type="file"),
    ],
    # ...
)


@demo_app.get("/from-mount")
def read_from_mount() -> dict:
    """Access the file directly at the mount path."""
    with open(DATA_MOUNT_PATH, "rb") as fh:
        return {"contents": fh.read().decode()}


@demo_app.get("/from-env-var")
def read_from_env_var() -> dict:
    """Access the file through its environment variable."""
    data_path = os.environ[DATA_ENV_VAR]
    with open(data_path, "rb") as fh:
        return {"contents": fh.read().decode()}


@demo_app.get("/from-get-parameter")
def read_from_get_parameter() -> dict:
    """Access the file using the get_parameter helper."""
    data_path = get_parameter("data")
    with open(data_path, "rb") as fh:
        return {"contents": fh.read().decode()}


@demo_app.get("/raw")
def read_raw() -> dict:
    """Access a parameter that has no mount or env_var."""
    data_path = get_parameter("data_raw")
    with open(data_path, "rb") as fh:
        return {"contents": fh.read().decode()}
# {{/docs-fragment parameter-access-methods}}

# {{docs-fragment parameter-serve-override}}
if __name__ == "__main__":
    import logging

    flyte.init_from_config(log_level=logging.DEBUG)

    app_handle = flyte.with_servecontext(
        parameter_values={
            demo_env.name: {
                "data": flyte.io.File.from_existing_remote("s3://bucket/data.txt"),
                "data_raw": flyte.io.File.from_existing_remote("s3://bucket/raw.txt"),
            }
        }
    ).serve(demo_env)
    print(f"Deployed app: {app_handle.url}")
# {{/docs-fragment parameter-serve-override}}

# {{docs-fragment runoutput-example}}
env = flyte.TaskEnvironment(name="training-env")

@env.task
async def train_model() -> flyte.io.File:
    # ... training logic ...
    return await flyte.io.File.from_local("/tmp/trained-model.pkl")

app_env4 = flyte.app.AppEnvironment(
    name="serving-app",
    parameters=[
        Parameter(
            name="model",
            value=flyte.app.RunOutput(type="file", task_name="training-env.train_model"),
            mount="/app/model",
        ),
    ],
    # ...
)
# {{/docs-fragment runoutput-example}}

# {{docs-fragment appendpoint-example}}
app1_env = flyte.app.AppEnvironment(name="backend-api")

app2_env = flyte.app.AppEnvironment(
    name="frontend-app",
    parameters=[
        Parameter(
            name="backend_url",
            value=flyte.app.AppEndpoint(app_name="backend-api"),
            env_var="BACKEND_URL",
        ),
    ],
    # ...
)
# {{/docs-fragment appendpoint-example}}

# {{docs-fragment runoutput-serving-example}}
import joblib
from sklearn.ensemble import RandomForestClassifier

training_env = flyte.TaskEnvironment(name="training-env")

@training_env.task
async def train_model_task() -> flyte.io.File:
    """Train a model and return it."""
    model = RandomForestClassifier()
    # ... training logic ...
    path = "./trained-model.pkl"
    joblib.dump(model, path)
    return await flyte.io.File.from_local(path)

serving_app = FastAPI()
model_serving_env = FastAPIAppEnvironment(
    name="model-serving-app",
    app=serving_app,
    parameters=[
        Parameter(
            name="model",
            value=flyte.app.RunOutput(
                type="file",
                task_name="training-env.train_model_task",
            ),
            mount="/app/model",
            env_var="MODEL_PATH",
        ),
    ],
)
# {{/docs-fragment runoutput-serving-example}}
