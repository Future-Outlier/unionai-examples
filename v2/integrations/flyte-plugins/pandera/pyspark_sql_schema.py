# /// script
# requires-python = ">=3.10"
# dependencies = [
#    "flyte>=2.0.0b52",
#    "flyteplugins-pandera",
#    "flyteplugins-spark",
#    "pandera[pyspark]",
# ]
# main = "main"
# ///

from __future__ import annotations

from typing import Annotated, cast

import pandera.typing.pyspark_sql as pt
import pyspark.sql.types as T
from flyteplugins.pandera import ValidationConfig
from flyteplugins.spark.task import Spark
from pandera.pyspark import DataFrameModel, Field
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import flyte

image = (
    flyte.Image.from_base("apache/spark-py:v3.4.0")
    .clone(name="pandera-pyspark-sql", python_version=(3, 10), extendable=True)
    .with_pip_packages(
        "flyteplugins-spark==2.0.9",
        "pandera[pyspark]",
    )
    .with_local_v2_plugins(["flyteplugins-spark", "flyteplugins-pandera"])
)

spark_conf = Spark(
    spark_conf={
        "spark.driver.memory": "1000M",
        "spark.executor.memory": "1000M",
        "spark.executor.cores": "1",
        "spark.executor.instances": "2",
        "spark.driver.cores": "1",
        "spark.kubernetes.file.upload.path": "/opt/spark/work-dir",
        "spark.jars": (
            "https://storage.googleapis.com/hadoop-lib/gcs/"
            "gcs-connector-hadoop3-latest.jar,"
            "https://repo1.maven.org/maven2/org/apache/hadoop/"
            "hadoop-aws/3.2.2/hadoop-aws-3.2.2.jar,"
            "https://repo1.maven.org/maven2/com/amazonaws/"
            "aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar"
        ),
    },
)

env = flyte.TaskEnvironment(
    name="pandera_pyspark_sql_schema",
    plugin_config=spark_conf,
    image=image,
    resources=flyte.Resources(cpu="1", memory="2Gi"),
)


# {{docs-fragment schemas}}
class EmployeeSchema(DataFrameModel):
    employee_id: int = Field(ge=0)
    name: str = Field()
    job_title: str = Field()


class EmployeeSchemaWithStatus(EmployeeSchema):
    status: str = Field(isin=["active", "inactive"])
# {{/docs-fragment}}


# {{docs-fragment build_valid_employees}}
@env.task(report=True)
async def build_valid_employees() -> pt.DataFrame[EmployeeSchema]:
    spark = cast(SparkSession, flyte.ctx().data["spark_session"])
    data = [
        (1, "Ada", "Engineer"),
        (2, "Grace", "Mathematician"),
        (3, "Barbara", "Computer scientist"),
    ]
    schema = T.StructType(
        [
            T.StructField("employee_id", T.IntegerType(), False),
            T.StructField("name", T.StringType(), False),
            T.StructField("job_title", T.StringType(), False),
        ]
    )
    return spark.createDataFrame(data, schema=schema)
# {{/docs-fragment}}


# {{docs-fragment pass_through}}
@env.task(report=True)
async def pass_through(
    df: pt.DataFrame[EmployeeSchema],
) -> pt.DataFrame[EmployeeSchemaWithStatus]:
    return df.withColumn("status", F.lit("active"))
# {{/docs-fragment}}


@env.task(report=True)
async def pass_through_with_error_warn(
    df: Annotated[
        pt.DataFrame[EmployeeSchema], ValidationConfig(on_error="warn")
    ],
) -> Annotated[
    pt.DataFrame[EmployeeSchemaWithStatus], ValidationConfig(on_error="warn")
]:
    return df.drop("name")


@env.task(report=True)
async def pass_through_with_error_raise(
    df: Annotated[
        pt.DataFrame[EmployeeSchema], ValidationConfig(on_error="warn")
    ],
) -> Annotated[
    pt.DataFrame[EmployeeSchemaWithStatus], ValidationConfig(on_error="raise")
]:
    return df.drop("name")


@env.task(report=True)
async def main() -> pt.DataFrame[EmployeeSchemaWithStatus]:
    df = await build_valid_employees()
    df2 = await pass_through(df)

    await pass_through_with_error_warn(df.drop("employee_id"))
    await pass_through_with_error_warn(df.withColumn("employee_id", F.lit(-1)))

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
    print("pyspark_sql pandera example OK:", run.outputs()[0])
