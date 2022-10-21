from fault_pkg import alarm_name2idx, nftype_name2idx
import dgl
import torch


class FaultGraphOnlyAlarmFeat(object):
    def __init__(self, fault_pkg):
        self.g = self.pkg2feat(fault_pkg)

    def pkg2feat(self, fault_pkg):
        pkg_alarm = set()
        for alarm in fault_pkg.ent_alarm.values():
            for a in alarm:
                pkg_alarm.add(a.alarm_name)

        alarm_feat = torch.zeros(len(alarm_name2idx))
        for a in list(pkg_alarm):
            alarm_feat[alarm_name2idx[a]] = 1

        return alarm_feat


class FaultGraph(object):
    def __init__(self, fault_pkg):
        self.g, self.ent_alarm_id_list = self.pkg2g(fault_pkg)

    def pkg2g(self, fault_pkg):
        global alarm_name2idx
        global nftype_name2idx
        self.fault_pkg = fault_pkg
        # src_ids = torch.tensor([tri[0] for tri in fault_pkg.tri_list] +
        #                        [tri[1] for tri in fault_pkg.tri_list])
        # dst_ids = torch.tensor([tri[1] for tri in fault_pkg.tri_list] +
        #                        [tri[0] for tri in fault_pkg.tri_list])
        src_ids = torch.tensor([tri[0] for tri in fault_pkg.tri_list])
        dst_ids = torch.tensor([tri[1] for tri in fault_pkg.tri_list])
        g = dgl.graph((src_ids, dst_ids))

        # alarm_feat
        alarm_feat = torch.zeros((g.num_nodes(), len(alarm_name2idx)))
        for k, v in fault_pkg.ent_alarm.items():
            for a in v:
                alarm_feat[k][alarm_name2idx[a.alarm_name]] += 1

        g.ndata['alarm_feat'] = alarm_feat

        # nftype
        nftype = torch.zeros(g.num_nodes(), dtype=torch.int64)
        for node_idx, ent in enumerate(fault_pkg.ent_list):
            nftype[node_idx] = nftype_name2idx[ent['nf_type']]
        g.ndata['nftype'] = nftype

        # ins_id_list
        ent_alarm_id_list = []
        for ent_idx, ent in enumerate(fault_pkg.ent_list):
            curr_ent_alarm_id = []
            if ent_idx in fault_pkg.ent_alarm:
                for alarm in fault_pkg.ent_alarm[ent_idx]:
                    curr_ent_alarm_id.append(fault_pkg.pkg_name + '&&' + alarm.alarm_topo_id)
            ent_alarm_id_list.append(curr_ent_alarm_id)

        return g, ent_alarm_id_list


# class FaultGraph(object):
#     def __init__(self, fault_pkg):
#         self.g = self.pkg2g(fault_pkg)
#
#     def pkg2g(self, fault_pkg):
#         global alarm_name2idx
#         global nftype_name2idx
#
#         src_ids = torch.tensor([tri[0] for tri in fault_pkg.tri_list] +
#                                [tri[1] for tri in fault_pkg.tri_list])
#         dst_ids = torch.tensor([tri[1] for tri in fault_pkg.tri_list] +
#                                [tri[0] for tri in fault_pkg.tri_list])
#         g = dgl.graph((src_ids, dst_ids))
#
#         alarm_node_idx = g.num_nodes()
#         alarm_idx_feat = []
#         add_src = []
#         add_dst = []
#         for k, v in fault_pkg.ent_alarm.items():
#             for alarm in v:
#                 add_src.append(alarm_node_idx)
#                 add_dst.append(k)
#                 alarm_idx_feat.append(alarm_name2idx[alarm.alarm_name])
#                 alarm_node_idx += 1
#         g = dgl.add_edges(g, add_src, add_dst)
#         alarm_idx_feat = torch.cat([torch.zeros(len(fault_pkg.ent_list), dtype=torch.int64) - 1,
#                                     torch.LongTensor(alarm_idx_feat)])
#         g.ndata['alarm_idx'] = alarm_idx_feat
#
#         nftype_idx_feat = torch.zeros(g.num_nodes(), dtype=torch.int64) - 1
#         for node_idx, ent in enumerate(fault_pkg.ent_list):
#             nftype_idx_feat[node_idx] = nftype_name2idx[ent['nf_type']]
#         g.ndata['nftype_idx'] = nftype_idx_feat
#
#         return g

if __name__ == '__main__':
    data_path = './data/团泊洼实验室数据'
    pkg_list = []
    for pkg_path in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, pkg_path)):
            pkg = FaultPKG(os.path.join(data_path, pkg_path))
            if pkg.has_mdaf and pkg.statistic['num_ent'] != 0 and \
                    pkg.statistic['num_alarm'] != 0 and pkg.statistic['label'] != '':
                pkg_list.append(pkg)

    g_list = [FaultGraph(pkg) for pkg in pkg_list]