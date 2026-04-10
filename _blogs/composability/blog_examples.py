"""
Blog: Framework Primitives vs Composability

Four runnable examples demonstrating Union.ai's composable approach
to distributed computing. Each example maps to a section in the blog post.

Run any example:
    python blog_examples.py broadcast_join
    python blog_examples.py compose_steps
    python blog_examples.py progressive_batching
    python blog_examples.py cached_groupby
"""

import asyncio
import sys
import time
import random
import pandas as pd
from typing import Dict, List

import flyte

env = flyte.TaskEnvironment(
    name="blog_examples",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "pandas",
        "pyarrow",
    ),
)

# ---------------------------------------------------------------------------
# Example 1: Broadcast Join (Blog Section 1)
#
# Instead of a distributed shuffle via .join(), load a small dimension
# table into memory and broadcast it to all workers.
# ---------------------------------------------------------------------------

# Load dimension table once, share across all workers
CITIES = pd.DataFrame({
    "city_id": [1, 2, 3, 4, 5],
    "city_name": ["New York", "London", "Tokyo", "Berlin", "Sydney"],
    "country": ["US", "UK", "JP", "DE", "AU"],
}).set_index("city_id")


def _make_records(chunk_id: int) -> pd.DataFrame:
    """Generate sample records for a chunk."""
    n = 5
    return pd.DataFrame({
        "record_id": range(chunk_id * n, chunk_id * n + n),
        "city_id": [random.randint(1, 5) for _ in range(n)],
        "value": [random.randint(10, 100) for _ in range(n)],
    })


@env.task
async def enrich_with_city(records: pd.DataFrame) -> pd.DataFrame:
    """Custom join using in-memory broadcast instead of distributed shuffle"""
    # In-memory merge - no network shuffle needed
    return records.join(CITIES, on="city_id", how="inner")


@env.task
async def process_all_chunks(num_chunks: int = 4) -> List[pd.DataFrame]:
    """Orchestrate distributed enrichment"""
    data_chunks = [_make_records(i) for i in range(num_chunks)]

    # Fan out with your custom logic
    results = await asyncio.gather(*[
        enrich_with_city(chunk)
        for chunk in data_chunks
    ])

    for i, df in enumerate(results):
        print(f"Chunk {i}: {len(df)} enriched records")
    return results


# ---------------------------------------------------------------------------
# Example 2: Compose Steps (Blog Section 2a)
#
# Instead of three sequential fan-out phases with wait points,
# compose pull + transform + enrich into ONE distributed task.
# ---------------------------------------------------------------------------

CATEGORIES = pd.DataFrame({
    "category_id": [1, 2, 3],
    "category_name": ["Electronics", "Clothing", "Food"],
}).set_index("category_id")


def query_source(source_id: int) -> pd.DataFrame:
    """Simulate pulling data from a source."""
    time.sleep(random.uniform(0.5, 2.0))  # Variable latency
    n = 3
    return pd.DataFrame({
        "id": range(source_id * n, source_id * n + n),
        "city_id": [random.randint(1, 5) for _ in range(n)],
        "category_id": [random.randint(1, 3) for _ in range(n)],
        "amount": [random.randint(10, 500) for _ in range(n)],
    })


def apply_business_logic(records: pd.DataFrame) -> pd.DataFrame:
    """Apply transformations."""
    records["amount_adjusted"] = records["amount"] * 1.1
    return records


@env.task
async def pull_transform_enrich(source_id: int) -> pd.DataFrame:
    """
    Compose pull + transform + enrich into ONE distributed task.
    No framework primitives. No waiting between steps.
    """
    # Pull from source
    records = query_source(source_id)

    # Transform immediately
    transformed = apply_business_logic(records)

    # Enrich immediately (in-memory joins, no shuffle)
    enriched = (
        transformed
        .join(CITIES, on="city_id", how="inner")
        .join(CATEGORIES, on="category_id", how="inner")
    )

    return enriched


@env.task
async def process_all_sources(num_sources: int = 6) -> List[pd.DataFrame]:
    """Orchestrate distributed processing"""
    # Fan out once over the composed operation
    results = await asyncio.gather(*[
        pull_transform_enrich(source_id)
        for source_id in range(num_sources)
    ])

    for i, df in enumerate(results):
        print(f"Source {i}: {len(df)} records processed")
    return results


# ---------------------------------------------------------------------------
# Example 3: Progressive Batch Processing (Blog Section 2b)
#
# Use asyncio.as_completed() to process results as they finish
# rather than waiting for all tasks.
# ---------------------------------------------------------------------------

@env.task
async def heavy_processing(item: int) -> int:
    """Simulate heavy processing with variable latency."""
    await asyncio.sleep(random.uniform(0.1, 3.0))
    return item * 2


@env.task
async def reduce_batch(batch: List[int]) -> int:
    """Reduce a batch of results."""
    return sum(batch)


def final_reduce(batch_results: List[int]) -> int:
    """Combine reduced batches."""
    return sum(batch_results)


@env.task
async def process_and_reduce_progressively(
    items: List[int],
    batch_size: int = 5,
) -> int:
    # Fan out to process all items
    process_tasks = [
        asyncio.create_task(heavy_processing(item))
        for item in items
    ]

    # Don't wait for all! Process batches as they complete
    reduce_tasks = []
    batch = []

    for task in asyncio.as_completed(process_tasks):
        result = await task
        batch.append(result)

        # Start reducing as soon as batch is full
        if len(batch) >= batch_size:
            reduce_tasks.append(
                asyncio.create_task(reduce_batch(batch.copy()))
            )
            batch.clear()

    # Handle remaining items
    if batch:
        reduce_tasks.append(asyncio.create_task(reduce_batch(batch)))

    # Combine reduced batches
    batch_results = await asyncio.gather(*reduce_tasks)
    return final_reduce(batch_results)


# ---------------------------------------------------------------------------
# Example 4: Cached Groupby (Blog Section 3)
#
# Each group's reduce is cached independently with cache="auto".
# If the workflow fails after 40k of 50k groups, completed ones
# don't re-run.
# ---------------------------------------------------------------------------

def expensive_computation(values: List[int]) -> int:
    """Simulate expensive computation."""
    time.sleep(1)
    return sum(values)


@env.task(cache="auto")  # Cache each reduce independently
async def reduce_group(key: str, values: List[int]) -> int:
    """Custom reduce with caching - survives failures"""
    return expensive_computation(values)


@env.task
async def process_with_cached_reduces(data: Dict[str, List[int]]) -> Dict[str, int]:
    # Fan out to 50k+ groups, each cached independently
    keys = list(data.keys())
    tasks = [reduce_group(key, data[key]) for key in keys]

    # Run all groups in parallel
    values = await asyncio.gather(*tasks)

    # Combine keys and values into result dict
    results = dict(zip(keys, values))
    return results


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

EXAMPLES = {
    "broadcast_join": (
        process_all_chunks,
        {"num_chunks": 4},
    ),
    "compose_steps": (
        process_all_sources,
        {"num_sources": 6},
    ),
    "progressive_batching": (
        process_and_reduce_progressively,
        {"items": list(range(20)), "batch_size": 5},
    ),
    "cached_groupby": (
        process_with_cached_reduces,
        {"data": {
            "USA": [100, 200, 150, 300],
            "Canada": [50, 75, 125],
            "UK": [200, 250, 100],
            "Germany": [150, 200, 175, 225],
            "France": [100, 150, 200],
        }},
    ),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in EXAMPLES:
        print(f"Usage: python {sys.argv[0]} <example>")
        print(f"Examples: {', '.join(EXAMPLES.keys())}")
        sys.exit(1)

    name = sys.argv[1]
    fn, kwargs = EXAMPLES[name]

    flyte.init_from_config("/Users/danielsola/repos/flyte-sdk/examples/config.yaml")
    print(f"Running: {name}")
    run = flyte.run(fn, **kwargs)
    print(f"Run URL: {run.url}")
