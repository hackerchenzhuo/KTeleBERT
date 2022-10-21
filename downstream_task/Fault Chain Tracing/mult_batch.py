import os
from unittest import result
import pandas as pd
import main
from collections import defaultdict as ddict
def mult_batch():
    result = ddict(list)
    work_path = ['yz_1020']

    path1 = os.getcwd()
    path2 = os.path.join(path1, 'embedding')
    for current_work_path in work_path:
        path3 = os.path.join(path2, current_work_path)
        for st in os.listdir(path3):
            if 'withAttributeAndNet' in st:
                arg = dict()
                arg['pretrain_all_path'] = os.path.join(path3, st)
                output = main.main(arg)
                output = dict(output[0])
                
                result['PretrainData'].append(st)
                for key, value in output.items():
                    result[key].append(round(value, 3))

    df = pd.DataFrame(result)
    df = df.set_index('PretrainData')
    df.to_excel('result_yz_1020.xlsx')
    print("done!")

if __name__ == '__main__':
    mult_batch()