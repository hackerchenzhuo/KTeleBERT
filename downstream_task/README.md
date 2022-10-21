# Fault Chain Tracing

## Workflow

<div align=center>
<center class="half">
    <img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/Fault%20Chain%20Tracing.png" height="350px"/>
</center></div>

## Usage

### Requirements
- `pytorch_lightning==1.5.10`
- `PyYAML>=6.0`
- `wandb>=0.12.7`
- `dgl>=0.7.1`
- `neuralkg>=1.0.21`

### Parameter
For more details: ```./neuralkg/utils/setup_parser.py```
```
 --model_name
 --dataset_name
 --data_path
 --litmodel_name
 --max_epochs
 --emb_dim
 --train_bs
 --eval_bs
 --num_neg
 --check_per_epoch
```

### Running

- train and test:
```bash
sh scripts/Network/GTransE_Rule.sh

sh scripts/Network/GTransE_Net_filter.sh
```

**Note**: 
The model of fault chain tracing is implemented with PyTorch and framework NeuralKG.


# Event Association Prediction

## Workflow

<div align=center>
<center class="half">
    <img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/Event%20Association%20Prediction.png" height="350px"/>
</center></div>

## Usage

### Requirements

- `torch`
- `sklearn`
- `tqdm`

### Parameter

- EPOCH = 15
- TEXT_DIM = 768
- NODE_NUM = 32
- NODE_DIM = 16
- TIME_DIM = 4
- BATCH_SIZE = 64
- LR = 0.001
- SEED = 2022
- N_FOLD= 5


### Running

- train & test 
```bash
python run_downstream.py

# Root Cause Analysis

## Workflow

<div align=center>
<center class="half">
    <img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/Root%20Cause%20Analysis.png" height="350px"/>
</center></div>

## Usage

### Requirements

- `torch == 1.12.0`
- `numpy == 1.22.3`
- `dgl == 0.8.2`


### Parameter

- num_fold = 5
- use_rule = 'count'
- withAttr = 'True
- num_epoch = 1000
- nlayer = 2
- train_bs = 99
- lr = 0.001
- early_stop_patience = 30


### Running

- train & test

```bash
bash run.sh
