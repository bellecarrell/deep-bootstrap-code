# deep-bootstrap-code

Code for the paper [The Deep Bootstrap Framework: Good Online Learners are Good Offline Generalizers](https://arxiv.org/abs/2010.08127).

The main training code is [here](/inftrain/train.py), and a sample configuration of hyperparameter sweep (using [Caliban](https://github.com/google/caliban)) is [here](/inftrain/sample_sweep.json).

The CIFAR-5m dataset is released at: https://github.com/preetum/cifar5m

## Expanse

Adapting codebase to use a local filesystem for logging and model storage, etc.

Run command: `python -m inftrain.train --proj <wandb_proj_name>`

CIFAR-5m dataset in `*.npz` format is ~17.2GB so I recommend taking a machine with 32-64GB memory,
depending on what batch size and network you want to run with.

Newly added args:
```
--save_model_step {freq}: how frequent (in batches) to save intermediate models
```

### Fine-tuning

`python -m inftrain.finetune --proj finetune-test --wandb_mode offline --k 4 --pretrained /path/to/model.pt --aug 4`
