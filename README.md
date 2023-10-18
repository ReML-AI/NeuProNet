# NeuProNet: Neural Profiling Networks for Sound Classification
The official repository for the paper "NeuProNet: Neural Profiling Networks for Sound Classification".

## Overview
>Real-world sound signals exhibit various aspects of grouping and profiling behaviors, such as being recorded from identical sources, having similar environmental settings, or encountering same background noises. In this work, we propose novel neural profiling networks (NeuProNet) capable of learning and extracting high-level unique profile representations from sounds. An end-to-end framework is developed so that any backbone architectures can be plugged in and trained, achieving better performance in any downstream sound classification tasks. We introduce an in-batch profile grouping mechanism based on profile awareness and attention pooling to produce reliable and robust features with contrastive learning. Furthermore, extensive experiments are conducted on multiple benchmark datasets and tasks to show that neural computing models under the guidance of our framework achieve significant performance gaps across all evaluation tasks. Particularly, the integration of NeuProNet surpasses recent state-of-the-art (SoTA) approaches on UrbanSound8K and VocalSound dataset with statistically significant improvements in benchmarking metrics, up to 5.92% in accuracy compared to the previous SoTA method and up to 20.19% compared to baselines. Our work provides a strong motivation for utilizing neural profiling for sound-related tasks.

## About this implementation

This implementation is based on the [fairseq toolkit](https://github.com/facebookresearch/fairseq).
Our main source code was written in the following directory:
+ [Spectrum Data Loader](fairseq/data)
+ [NeuProNet framework and backbones implementation](fairseq/models/NeuProNet)
+ [NeuProNet loss function](fairseq/criterions/NeuProNet_criterion.py)
+ [Configuration files for training baselines with or without NeuProNet](examples/NeuProNet)

## Requirements and Installation
Please follow the instructions to [install the framework](https://github.com/facebookresearch/fairseq#getting-started).

Additionally, install librosa and efficientnet_pytorch:
```
pip install librosa soundfile efficientnet_pytorch
```

## Training

Without NeuProNet:
```
fairseq-hydra-train task.data=$DATA_PATH task.profiling=False --config-dir examples/NeuProNet/config/finetuning --config-name urbansound8k

fairseq-hydra-train task.data=$DATA_PATH task.profiling=False --config-dir examples/NeuProNet/config/finetuning --config-name vocalsound
```

With NeuProNet:
```
fairseq-hydra-train task.data=$DATA_PATH task.profiling=True --config-dir examples/NeuProNet/config/finetuning --config-name urbansound8k

fairseq-hydra-train task.data=$DATA_PATH task.profiling=True --config-dir examples/NeuProNet/config/finetuning --config-name vocalsound
```

## Evaluation

For models trained on VocalSound
```
fairseq-validate --path $TRAINED_MODEL_PATH --task audio_finetuning $DATA_PATH --valid-subset test --batch-size 128
```

With NeuProNet:
```
fairseq-validate --path $TRAINED_MODEL_PATH --task audio_finetuning $DATA_PATH --valid-subset valid --batch-size 512
```
