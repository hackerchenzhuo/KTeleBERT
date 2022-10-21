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
