"""
Multi-node DeepSpeed fine-tuning tutorial for Qwen2.5-VL.

- `config.py`: constants, task environments, typed config
- `data.py`: dataset prep, dataset class, collator
- `model.py`: adapter module and Lightning module
- `callbacks.py`: live report, recovery, metrics callbacks
- `tasks.py`: training, evaluation, and pipeline tasks
- `utils.py`: misc utils
"""

import flyte

from config import (
    DEFAULT_CHECKPOINT_BASE_URI,
    DEFAULT_MODEL_NAME,
)
from tasks import qwen_vl_multinode_deepspeed
from flyte.io import Dir

# {{docs-fragment main-run}}
if __name__ == "__main__":
    flyte.init_from_config()

    run = flyte.run(
        qwen_vl_multinode_deepspeed,
        model_name=DEFAULT_MODEL_NAME,
        max_train_samples=512,
        max_val_samples=128,
        epochs=5,
        per_device_batch_size=1,
        target_global_batch_size=16,
        learning_rate=2e-4,
        reconstruction_loss_weight=0.35,
        eval_examples=16,
        checkpoint_base_uri=DEFAULT_CHECKPOINT_BASE_URI,
        wandb_project="qwen-vl-multinode-deepspeed",
        wandb_entity="<YOUR_WANDB_ENTITY>",  # TODO: update with your own wandb entity
        # resume_training_artifacts=Dir(
        #     path="s3://flyte-examples/qwen-vl-multinode-deepspeed/<ACTION_NAME>/qwen_vl_training_recovery/"
        # ),
    )

    print(f"Run URL: {run.url}")
# {{/docs-fragment}}
