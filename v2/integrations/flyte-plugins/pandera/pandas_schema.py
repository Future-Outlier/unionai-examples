# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte",
#    "flyteplugins-pandera",
#    "pandera[pandas]",
# ]
# main = "main"
# ///

from __future__ import annotations

from typing import Annotated

import pandas as pd
import pandera.pandas as pa
import pandera.typing.pandas as pt
from flyteplugins.pandera import ValidationConfig

import flyte

img = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "flyteplugins-pandera", "pandera[pandas]"
)

env = flyte.TaskEnvironment(
    "pandera_pandas_schema",
    image=img,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


class EmployeeSchema(pa.DataFrameModel):
    employee_id: int = pa.Field(ge=0)
    name: str


class EmployeeSchemaWithStatus(EmployeeSchema):
    status: str = pa.Field(isin=["active", "inactive"])


# {{docs-fragment build_valid_employees}}
@env.task(report=True)
async def build_valid_employees() -> pt.DataFrame[EmployeeSchema]:
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3],
            "name": ["Ada", "Grace", "Barbara"],
        }
    )
# {{/docs-fragment}}


# {{docs-fragment pass_through}}
@env.task(report=True)
async def pass_through(
    df: pt.DataFrame[EmployeeSchema],
) -> pt.DataFrame[EmployeeSchemaWithStatus]:
    return df.assign(status="active")
# {{/docs-fragment}}


# {{docs-fragment pass_through_with_error_warn}}
@env.task(report=True)
async def pass_through_with_error_warn(
    df: Annotated[
        pt.DataFrame[EmployeeSchema], ValidationConfig(on_error="warn")
    ],
) -> Annotated[
    pt.DataFrame[EmployeeSchemaWithStatus], ValidationConfig(on_error="warn")
]:
    del df["name"]
    return df
# {{/docs-fragment}}


# {{docs-fragment pass_through_with_error_raise}}
@env.task(report=True)
async def pass_through_with_error_raise(
    df: Annotated[
        pt.DataFrame[EmployeeSchema], ValidationConfig(on_error="warn")
    ],
) -> Annotated[
    pt.DataFrame[EmployeeSchemaWithStatus], ValidationConfig(on_error="raise")
]:
    del df["name"]
    return df
# {{/docs-fragment}}


@env.task(report=True)
async def main() -> pt.DataFrame[EmployeeSchemaWithStatus]:
    df = await build_valid_employees()
    df2 = await pass_through(df)

    await pass_through_with_error_warn(df.drop(["employee_id"], axis="columns"))
    await pass_through_with_error_warn(df.assign(employee_id=-1))

    try:
        await pass_through_with_error_raise(df)
    except Exception as exc:
        print(exc)

    return df2


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
    print("pandas pandera example OK:", run.outputs()[0])
