#data=0;
##rel=103;
#fact=2791;
#score=1000;
##for score in {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,100}
##  do
#  for rel in {3,5,7,10,15,20,25}
##      do
##      for fact in {2,3,5,10,15,20,25}
#        do
#        python joint_test.py --gpu_id 1 --exp_name fusion_prediction_zsl_rel --exp_id rel"${rel}"_fact"${fact}"data_"${data}"score_"${score}" --data_choice "${data}" --top_rel "${rel}" --top_fact "${fact}" --soft_score "${score}" --ZSL 1 --mrr 1
#        done
##      done
##  done


# python main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name tes \
#                --exp_id test_tiny \
#                --ernie_stratege -1 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpi \
#                --maxlength 256 \
#                --lr 1e-5 \
#                --ke_lr 3e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 1 \
#                --dist 0 \
#                --accumulation_steps 2 \
#                --freeze_layer 0 \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v20 \
#                --workers 12 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v41_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 256 \
#                --plm_emb_type cls \
#                --only_ke_loss 1 \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v21 \
#                --workers 12 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v41_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 256 \
#                --plm_emb_type cls \
#                --only_ke_loss 0 \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v202 \
#                --workers 12 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v4120_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 256 \
#                --plm_emb_type cls \
#                --only_ke_loss 0 \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v35 \
#                --workers 8 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 256 \
#                --plm_emb_type cls \
#                --only_ke_loss 0 \
#                --kg_data_name KG_data_base_rule \
#                --train_together 1 \
#                --mask_loss_scale 1.0 \
#             #    --neg_num 6 \

# _wholedata_without_specialToken

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 50 \
#                --exp_name Pre_train \
#                --exp_id 1017_v37 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_large \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \


python -m torch.distributed.launch --nproc_per_node=4 main.py --LLRD 1 \
               --eval_step 10 \
               --epoch 16 \
               --save_model 1 \
               --mask_stratege wwm \
               --batch_size 46 \
               --batch_size_ke 20 \
               --exp_name Fine_tune_2 \
               --exp_id 1010_v36 \
               --workers 8 \
               --use_NumEmb 1 \
               --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
               --maxlength 256 \
               --lr 4e-5 \
               --ke_lr 8e-5 \
               --train_strategy 2 \
               --model_name TeleBert2 \
               --train_ratio 1 \
               --save_pretrain 0 \
               --dist 1 \
               --accumulation_steps 8 \
               --accumulation_steps_ke 6 \
               --special_token_mask 0 \
               --freeze_layer 0 \
               --ernie_stratege -1 \
               --mlm_probability_increase curve \
               --use_kpi_loss 1 \
               --mlm_probability 0.4 \
               --use_awl 0 \
               --cls_head_init 1 \
               --emb_init 0 \
               --final_mlm_probability 0.4 \
               --ke_dim 256 \
               --plm_emb_type cls \
               --train_together 0 \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v32 \
#                --workers 8 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v4120_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 1 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 256 \
#                --plm_emb_type cls \
#                --only_ke_loss 0 \
#                --kg_data_name KG_data_base_rule \
#                --train_together 1 \
#                --mask_loss_scale 2.0 \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v33 \
#                --workers 8 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v41_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 256 \
#                --plm_emb_type cls \
#                --only_ke_loss 0 \
#                --kg_data_name KG_data_base_rule \
#                --train_together 1 \
#                --mask_loss_scale 1.0 \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v34 \
#                --workers 8 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v4120_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 256 \
#                --plm_emb_type last_avg \
#                --only_ke_loss 0 \
#                --kg_data_name KG_data_base_rule \
#                --train_together 1 \
#                --mask_loss_scale 1.0 \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v21 \
#                --workers 8 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v41_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 256 \
#                --plm_emb_type last_avg \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v22 \
#                --workers 8 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v41_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 768 \
#                --plm_emb_type cls \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v23 \
#                --workers 8 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v41_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 768 \
#                --plm_emb_type last_avg \

# python -m torch.distributed.launch --nproc_per_node=3 main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --batch_size_ke 28 \
#                --exp_name Fine_tune_2 \
#                --exp_id 1010_v24 \
#                --workers 8 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 2 \
#                --model_name Pre_train_0926_v4120_s42 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --accumulation_steps_ke 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 0 \
#                --emb_init 0 \
#                --final_mlm_probability 0.4 \
#                --ke_dim 256 \
#                --plm_emb_type cls \










# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 10 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0927_v44 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 1e-5 \
#                --ke_lr 2e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 1 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 3 \
#                --ernie_stratege 4 \
#                --mlm_probability_increase curve \
#                --special_token_mask 1 \
#                --use_kpi_loss 0 \
#             #    --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 10 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0927_v45 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 1e-5 \
#                --ke_lr 2e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 1 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 3 \
#                --ernie_stratege 4 \
#                --mlm_probability_increase curve \
#                --special_token_mask 1 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 10 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0927_v46 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 1e-5 \
#                --ke_lr 2e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 1 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 3 \
#                --ernie_stratege 4 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 10 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v42 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 1e-5 \
#                --ke_lr 2e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert3 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                # --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v411 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 1e-5 \
#                --ke_lr 2e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                # --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v413 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 1e-5 \
#                --ke_lr 2e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v414 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v415 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \


# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v417 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 0 \
#                --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v416 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 0 \
#                --mlm_probability 0.4 \

# -----------------

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v417 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 0 \
#                --mlm_probability 0.4 \


# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v401 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4121 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4120 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4119 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege 6 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4118 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \


# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4123 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \
#                --cls_head_init 0 \


# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4124 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 1 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \
#                --cls_head_init 1 \
#                --emb_init 1 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4125 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 0 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \
#                --cls_head_init 1 \
#                --emb_init 0 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4122 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 0 \
#                --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v402 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4124 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --special_token_mask 1 \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \
#                --cls_head_init 1 \
#                --emb_init 1 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 0 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4126 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \
# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4122 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 0 \
#                --mlm_probability 0.4 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4127 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \
#                --use_awl 0 \
            
# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4129 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \
#                --use_awl 0 \
#                --cls_head_init 1 \
#                --emb_init 0 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 16 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4130 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 1 \
#                --emb_init 0 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 32 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4131 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 8 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 1 \
#                --emb_init 0 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 24 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4132 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 4e-5 \
#                --ke_lr 8e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 6 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.4 \
#                --use_awl 0 \
#                --cls_head_init 1 \
#                --emb_init 0 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4127 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \
#                --use_awl 0 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v4128 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbwDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert2 \
#                --train_ratio 1 \
#                --save_pretrain 0 \
#                --dist 1 \
#                --accumulation_steps 3 \
#                --special_token_mask 0 \
#                --freeze_layer 3 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase curve \
#                --use_kpi_loss 1 \
#                --mlm_probability 0.15 \
#                --use_awl 1 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v32 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiDoc \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 1 \
#                --dist 1 \
#                --accumulation_steps 2 \
#                --special_token_mask 1 \
#                --freeze_layer 0 \
#                --ernie_stratege 2 \
#                --final_mlm_probability 0.15 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 12 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v22 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpiTbw \
#                --maxlength 256 \
#                --lr 2e-5 \
#                --ke_lr 4e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 1 \
#                --dist 1 \
#                --accumulation_steps 2 \
#                --special_token_mask 0 \
#                --freeze_layer 2 \
#                --ernie_stratege -1 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 15 \
#                --save_model 1 \
#                --mask_stratege rand \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v13 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpi \
#                --maxlength 256 \
#                --lr 1e-5 \
#                --ke_lr 2e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 1 \
#                --dist 1 \
#                --accumulation_steps 2 \
#                --special_token_mask 0 \
#                --freeze_layer 0 \
#                --ernie_stratege -1 \

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 15 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 64 \
#                --exp_name Pre_train \
#                --exp_id 0926_v14 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --seq_data_name Seq_data_RuAlmEntKpi \
#                --maxlength 256 \
#                --lr 1e-5 \
#                --ke_lr 2e-5 \
#                --train_strategy 1 \
#                --model_name TeleBert \
#                --train_ratio 1 \
#                --save_pretrain 1 \
#                --dist 1 \
#                --accumulation_steps 2 \
#                --special_token_mask 0 \
#                --freeze_layer 2 \
#                --ernie_stratege -1 \
#                --mlm_probability_increase linear \


# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 20 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 42 \
#                --exp_name order \
#                --exp_id tes_od_layer_1_cls \
#                --ernie_stratege -1 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --dist 1 \
#                --seq_data_name Seq_data_tiny \
#                --kg_data_name KG_data_base \
#                --maxlength 200 \
#                --lr 8e-6 \
#                --ke_lr 1e-4 \
#                --train_strategy 2 \
#                --model_name exp_0918_ke_768_DistributedDataParallel_id_p_2_seed_42_epoch_20 \
#                --num_od_layer 1 \
#                --plm_emb_type cls \
            #   --mask_stratege wwm \34

# python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --LLRD 1 \
#                --eval_step 10 \
#                --epoch 20 \
#                --save_model 1 \
#                --mask_stratege wwm \
#                --batch_size 42 \
#                --exp_name order \
#                --exp_id tes_od_layer_1_last_avg \
#                --ernie_stratege -1 \
#                --workers 6 \
#                --use_NumEmb 1 \
#                --dist 1 \
#                --seq_data_name Seq_data_tiny \
#                --kg_data_name KG_data_base \
#                --maxlength 200 \
#                --lr 8e-6 \
#                --ke_lr 1e-4 \
#                --train_strategy 2 \
#                --model_name exp_0918_ke_768_DistributedDataParallel_id_p_2_seed_42_epoch_20 \
#                --num_od_layer 1 \
#                --plm_emb_type last_avg \
            #   --mask_stratege wwm \34



            #   --mask_stratege wwm \34
# python main.py --only_test 1 \
#                --batch_size 32 \
#                --use_NumEmb 1 \
#                --mask_test 0 \
#                --embed_gen 1 \
#                --path_gen yz_data_whole5gc \
#                --mask_stratege wwm \
#                # --model_name exp_huawei_exp_HWBert_id_001_seed_42_epoch_15 \
#                # --embed_gen 1 \
# bash test.sh

