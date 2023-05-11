# KTeleBERT
![](https://img.shields.io/badge/version-1.0.0-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/hackerchenzhuo/KTeleBERT/blob/main/licence)
[![arxiv badge](https://img.shields.io/badge/arxiv-2210.11298-red)](https://arxiv.org/abs/2210.11298)

[*Tele-Knowledge Pre-training for Fault Analysis*](https://arxiv.org/abs/2210.11298)

>Author: Zhuo Chen†, Wen Zhang†, Yufeng Huang, Mingyang Chen,Yuxia Geng, Hongtao Yu, Zhen Bi, Yichi Zhang, Zhen Yao, Huajun Chen (College of Computer Science, **Zhejiang University**)
Wenting Song, Xinliang Wu, Yi Yang, Mingyi Chen, Zhaoyang Lian, Yingying Li, Lei Cheng (NAIE PDU, **Huawei Technologies** Co., Ltd.)
>In this paper we propose a tele-domain pre-trained language model named **TeleBERT** to learn the general semantic knowledge in the telecommunication field together with its improved version **KTeleBERT**, which incorporates those implicit information in machine log data and explicit knowledge contained in our Tele-product Knowledge Graph (Tele-KG).
## Tele-data

<div align=center><img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/TeleKG.png" height="440px"></div>


## Workflow
<div align=center>
<center class="half">
    <img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/workflow.png" height="200px"/><img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/Template.png" height="200px"/><img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/numericalEmbedding.png" height="200px"/>
</center></div>

## ANEnc
<div align=center>
<center class="half">
    <img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/ANEnc.png" height="350px"/>
</center></div>

## Visualization
- **Visualization for Numerical Data**
<div align=center>
<center class="half">
    <img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/vis.png" height="250px"/>
</center></div>

- **Visualization for Abnormal KPI Detection Data**
<div align=center>
<center class="half">
    <img src="https://github.com/hackerchenzhuo/KTeleBERT/blob/main/figures/ad vis.png" height="500px"/>
</center></div>

## Usage

### Requirements
- `transformers >= 4.21.2`
- `PyTorch >= 1.6.0`
- `tqdm`
- `ltp`

### Parameter
For more details: ```config.py```
```
  --train_strategy 
  --batch_size 
  --batch_size_ke 
  --batch_size_od 
  --batch_size_ad 
  --epoch 
  --save_model {0,1}
  --save_pretrain {0,1}
  --from_pretrain {0,1}
  --dump_path   Experiment dump path
  --random_seed 
  --train_ratio  ratio for train/test
  --final_mlm_probability 
  --mlm_probability_increase {linear,curve}
  --mask_stratege {rand,wwm,domain}
  --ernie_stratege 
  --use_mlm_task {0,1}
  --add_special_word {0,1}
  --freeze_layer {0,1,2,3,4}
  --special_token_mask {0,1}
  --emb_init {0,1}
  --cls_head_init {0,1}
  --use_awl {0,1}
  --mask_loss_scale 
  --ke_norm 
  --ke_dim 
  --ke_margin 
  --neg_num 
  --adv_temp    The temperature of sampling in self-adversarial negative sampling.
  --ke_lr 
  --only_ke_loss 
  --use_NumEmb 
  --contrastive_loss {0,1}
  --l_layers L_LAYERS
  --use_kpi_loss
  --only_test {0,1}
  --mask_test {0,1}
  --embed_gen {0,1}
  --ke_test {0,1}
  --ke_test_num 
  --path_gen 
  --order_load 
  --order_num 
  --od_type {linear_cat,vertical_attention}
  --eps EPS             label smoothing
  --num_od_layer 
  --plm_emb_type {cls,last_avg}
  --order_test_name 
  --order_threshold 
  --rank RANK           rank to dist
  --dist DIST           whether to dist
  --device DEVICE       device id (i.e. 0 or 0,1 or cpu)
  --world-size WORLD_SIZE number of distributed processes
  --dist-url DIST_URL   url used to set up distributed training
  --local_rank LOCAL_RANK
```


### Running

- train:
```bash run.sh```
``` ```
- test:
```bash test.sh```
``` ```

**Note**: 
- you can open the `.sh` file for default parameter modification.

<a href="https://info.flagcounter.com/VOlE"><img src="https://s11.flagcounter.com/count2/VOlE/bg_FFFFFF/txt_000000/border_F7F7F7/columns_6/maxflags_12/viewers_3/labels_0/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
