FILES=pretrain/*
keyword='ent'
for run in {1..3}
do
    for f in $FILES
    do
        for rule in 'count' 'no'
        do
            model=${f##*/}
            withAttr='False'
            task_group_name=${model}_${rule}_${keyword}_${run}
            pt_emb_path=./pretrain/${model}
            python main_multifold.py --use_rule ${rule} --pt_emb_path ${pt_emb_path} --task_group_name ${task_group_name} --withAttr ${withAttr}
            rm -r state
        done
    done
done
python handleLog.py --keyword ${keyword}    