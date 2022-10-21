from fault_pkg import get_pkg_list
import numpy as np
import pickle as pkl

train_data_path = './data/团泊洼实验室数据'
valid_data_path = './data/团泊洼实验室数据测试集'

train_pkg_list = get_pkg_list(train_data_path, None)
valid_pkg_list = get_pkg_list(valid_data_path, None)

pkg_list = train_pkg_list + valid_pkg_list

pkg_idx = np.arange(len(pkg_list))
np.random.shuffle(pkg_idx)

pkg_list_rand = [pkg_list[i].pkg_path for i in pkg_idx]
pkl.dump(pkg_list_rand, open('./data/rand_pkg_path_yht.pkl', 'wb'))