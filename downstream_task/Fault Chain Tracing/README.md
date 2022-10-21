# 数据集介绍

entities.dict 实体映射字典,建立了id到entity的字典 entity是以“alarm / kpi”的形式作为主元
```bash
id /t entity
```

relations.dict 关系映射字典,建立了id到relation的字典 relation是以“网元1_网元2”的形式作为主元
```bash
id /t relation
```

train.txt 训练集, 收集到的规则 / 根据规则点亮的单跳三元组
```
head /t relation /t tail /t confidence
```

valid.txt 验证集, 收集到的规则 / 根据规则点亮的两跳三元组
```
RuleKG: 收集到的规则
head /t relation /t tail /t confidence

network: 根据规则点亮的两跳三元组
head /t relation1 /t tail1 /t confidence1 /t relation2 /t tail2 /t confidence2
```

test.txt  测试集, 收集到的规则 / 根据规则点亮的两跳三元组
```
RuleKG: 收集到的规则
head /t relation /t tail /t confidence

network: 根据规则点亮的两跳三元组
head /t relation1 /t tail1 /t confidence1 /t relation2 /t tail2 /t confidence2
```

# 运行说明
在收集到规则全集中进行预训练
```bash
sh scripts/Network/GTransE_Rule.sh
```
将预训练模型应用到Network数据集中进行测试(需要设置参数RULEKE_PATH为预训练模型路径)
```bash
sh scripts/Network/GTransE_Net_filter.sh
```

# 测试结果
线上可视化测试结果: wandb网站（https://wandb.ai/home）

离线记录logging测试结果: loggging文件夹

模型生成路径: output文件夹
