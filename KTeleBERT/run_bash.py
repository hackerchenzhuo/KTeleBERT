import os
os.system('bash run.sh')

# 顺序：
# copy_data.sh
# data_trans.py

# 如果修改了special token 需要从这里开始运行 
# get_chinese_ref.py
# special_token_pre_emb.py