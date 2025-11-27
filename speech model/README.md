# Speech Model (wav2vec 2.0)

This guide provides instructions for reproducing the wav2vec 2.0 speech recognition experiments as presented in our paper. We provide implementations with Derf (our proposed function), DyT, and LayerNorm. Follow the steps below to set up the environment, train the model, and evaluate the results.


## Installation
Set up the Python environment with the following commands:
```
conda create -n w2v python=3.10
conda activate w2v
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install soundfile

cd fairseq
pip install --editable ./
```

## Training & Evaluation
To train and evaluate the ViT models on the LibriSpeech dataset, run the following commands:

### wav2vec 2.0 Base

```
torchrun --nnodes=8 --nproc_per_node=8 fairseq-hydra-train \
    task.data=/path/to/manifest \
    --config-dir ./examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech \
    --normtype $NORMTYPE
```

### wav2vec 2.0 Large

```
torchrun --nnodes=16 --nproc_per_node=8 fairseq-hydra-train \
    task.data=/path/to/manifest \
    --config-dir ./examples/wav2vec/config/pretraining \
    --config-name wav2vec2_large_librispeech \
    --normtype $NORMTYPE
```

- Replace `$NORMTYPE` to choose which point-wise function or normalization layer to use. Available options include: `derf` (our proposed function), `dyt` or `layernorm` (DyT or LayerNorm as baselines).

- For further details about wav2vec 2.0, see the [original repository](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md).