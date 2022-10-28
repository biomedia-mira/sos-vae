# Structured Uncertainty in the Observation Space of Variational Autoencoders

This repository contains the code for the paper
> J. Langley, M. Monteiro, C, Jones, N. Pawlowski, B. Glocker. _Structured Uncertainty in the Observation Space of Variational Autoencoders_. Transactions on Machine Learning Research. 2022 [[OpenReview]](https://openreview.net/forum?id=cxp7n9q5c4)

## Installation

A virtual environment is recommended for this project. Create and activate a virtual environment as follows
```bash
python3 -m venv venv
source venv/bin/activate
```
Install required packages:
```bash
pip install -r requirements.txt
```

### Caveat

If running fails with a `CUDA Runtime Error`: make sure you have installed the correct torch binary compiled with your CUDA version.

* You can find the CUDA version by running
  ```bash
  nvidia-smi
  ```
* You can find the version of CUDA torch was compiled with by running the following python
  ```python
  import torch
  torch.version.cuda
  ```

If the do not match (at least in major version number), you will need to install the correct version of torch (and torchvision).

To do so [run the following command](https://discuss.pytorch.org/t/old-cuda-driver-with-pytorch-1-5/78470/4):
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
With the correct torch (and torchvision) versions *and* and the right CUDA-compiled version.
For example, the above command installs torch and torchvision versions 1.8.0 and 0.9.0 respectively **with CUDA-compiled 11.1**.

Now check `torch.version.cuda` matches (in at least major version number) `nvidia-smi`.

Good to go!

## Training

```bash
python main.py
```

For options run `python src/main.py --help` to show the following:

```
usage: main.py [-h] [-g] [-f64] [-dnv] [-d] [-t] [-s] [-e EPOCHS]
               [-b BATCH_SIZE] [-lr LEARNING_RATE] [-f LATENT_FEATURES]
               [-r RANK] [-kl TARGET_KL] [-var TARGET_VAR] [-p PATCH_SIZE]
               [-pre PRE_TRAIN] [--grayscale] [-cp CHECKPOINTS]
               [--resume RESUME]

Train and evaluate network

optional arguments:
  -h, --help            show this help message and exit
  -g, --gpu             Evaluate using GPU if available
  -f64, --float64       Evaluate using 64-bit floats. 32 otherwise
  -dnv, --dist-no-val   Disable distribution validation checks
  -d, --deterministic   Set deterministic/repeatable computation
  -t, --title           Enable title prompt
  -s, --save            Save model_dict upon run completion
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train for
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -f LATENT_FEATURES, --latent-features LATENT_FEATURES
                        Dimensionality of latent space
  -r RANK, --rank RANK  Rank of the decoder covariance
  -kl TARGET_KL, --target-kl TARGET_KL
                        Target KL loss (per batch item)
  -var TARGET_VAR, --target-var TARGET_VAR
                        Target mean variance of the output distribution
  -p PATCH_SIZE, --patch-size PATCH_SIZE
                        Size of the ground truth patch used for correction in the visualiser
  -pre PRE_TRAIN, --pre-train PRE_TRAIN
                        Number of epochs to pre-train the model for
  --grayscale           Run in grayscale
  -cp CHECKPOINTS, --checkpoints CHECKPOINTS
                        Epoch interval between checkpoints. No checkpoints
                        otherwise
  --resume RESUME       ID of a previous run to resume running (if specified).
                        Resuming with modified args should be attempted with
                        caution
```

## Checkpoints

Use the above described `--checkpoints` argument, this program will save checkpoints at a user-specified interval of epochs.
Resuming a checkpointed run can be easily done by specifying the `--resume` argument with the respective ID.

Please note that resuming a checkpointed run does not *remember* the previous program arguments and should be called with the same arguments.
Resuming a run with different arguments has been implemented, but should only be attempted with caution and expertise.

When a run that has been checkpointed finishes the final time, it is possible to make tensorboard see the individual checkpointed runs as one continuous run.
To do this, combine the `events.out.tfevents....` under one subdirectory within the `runs` folder.

## Logging

This project uses tensorboard for all logging
```bash
tensorboard --logdir runs --samples_per_plugin images=9999
```
