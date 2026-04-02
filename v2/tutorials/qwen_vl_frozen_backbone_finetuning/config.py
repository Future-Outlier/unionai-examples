import flyte
from dataclasses import asdict, dataclass
from typing import NamedTuple

from flyte.io import Dir
from flyteplugins.jsonl import JsonlFile
from flyteplugins.pytorch.task import Elastic

# {{docs-fragment topology}}
NUM_NODES = 2
DEVICES_PER_NODE = 4
IMAGE_SIZE = 224
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# {{/docs-fragment topology}}
DEFAULT_CHECKPOINT_BASE_URI = (
    "s3://flyte-examples/qwen-vl-multinode-deepspeed"  # TODO: update with your own URI
)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# {{docs-fragment gpu-image}}
gpu_image = (
    flyte.Image.from_base("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04")
    .clone(name="qwen_vl_multinode_deepspeed", python_version=(3, 13), extendable=True)
    .with_apt_packages("build-essential")
    .with_pip_packages(
        "torch==2.9.1",
        "torchvision==0.24.1",
        "lightning==2.6.1",
        "transformers==4.57.3",
        "deepspeed==0.18.8",
        "datasets==4.4.1",
        "pillow==11.3.0",
        "flyteplugins-pytorch>=2.0.11",
        "flyteplugins-jsonl>=2.0.11",
        "flyteplugins-wandb>=2.0.11",
    )
)
# {{/docs-fragment gpu-image}}

# {{docs-fragment non-gpu-image}}
non_gpu_image = flyte.Image.from_debian_base(
    name="qwen_vl_multinode_deepspeed_non_gpu"
).with_pip_packages(
    "flyteplugins-pytorch>=2.0.11",
    "flyteplugins-jsonl>=2.0.11",
    "flyteplugins-wandb>=2.0.11",
    "lightning==2.6.1",
    "datasets==4.4.1",
    "pillow==11.3.0",
    "torchvision==0.24.1",
)
# {{/docs-fragment non-gpu-image}}

# {{docs-fragment task-environments}}
dataset_env = flyte.TaskEnvironment(
    name="qwen_vl_dataset_prep",
    image=non_gpu_image,
    resources=flyte.Resources(cpu=5, memory="15Gi"),
    cache="auto",
)

training_env = flyte.TaskEnvironment(
    name="qwen_vl_multinode_training",
    image=gpu_image,
    resources=flyte.Resources(
        cpu=42,
        memory="256Gi",
        gpu=f"L40s:{DEVICES_PER_NODE}",
        shm="16Gi",
    ),
    plugin_config=Elastic(nnodes=NUM_NODES, nproc_per_node=DEVICES_PER_NODE),
    secrets=[
        flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY")
    ],  # TODO: update with your own secret key
    env_vars={
        "TORCH_DISTRIBUTED_DEBUG": "INFO",
        "NCCL_DEBUG": "WARN",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_HOME": "/usr/local/cuda",
        "DS_SKIP_CUDA_CHECK": "1",
    },
    cache="auto",
)

evaluation_env = flyte.TaskEnvironment(
    name="qwen_vl_adapter_eval",
    image=gpu_image,
    resources=flyte.Resources(cpu=16, memory="64Gi", gpu="L40s:1"),
    cache="auto",
)

driver_env = flyte.TaskEnvironment(
    name="qwen_vl_multinode_driver",
    image=non_gpu_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[dataset_env, training_env, evaluation_env],
)
# {{/docs-fragment task-environments}}


# {{docs-fragment config-dataclass}}
@dataclass
class Config:
    model_name: str = DEFAULT_MODEL_NAME
    image_size: int = IMAGE_SIZE
    max_train_samples: int = 1024
    max_val_samples: int = 256
    epochs: int = 8
    per_device_batch_size: int = 1
    target_global_batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    reconstruction_loss_weight: float = 0.35
    report_every_n_steps: int = 10
    num_workers: int = 4
    max_length: int = 512
    eval_examples: int = 16
    train_occlusion_min: float = 0.22
    train_occlusion_max: float = 0.42
    eval_occlusion_min: float = 0.28
    eval_occlusion_max: float = 0.45
    seed: int = 7

    def to_dict(self) -> dict:
        return asdict(self)
# {{/docs-fragment config-dataclass}}


class DatasetArtifacts(NamedTuple):
    train_manifest: JsonlFile
    val_manifest: JsonlFile
    images: Dir
