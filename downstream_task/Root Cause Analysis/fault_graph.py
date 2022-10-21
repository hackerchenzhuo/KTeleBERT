import dgl
import torch
import json
import pickle as pkl


class FaultGraph(object):
    def __init__(self, fault_pkg):
        self.fault_pkg = fault_pkg
        self.args = fault_pkg.args

        map_dict = pkl.load(open(self.args.map_dict_path, 'rb'))
        self.alarm_name2idx = map_dict['alarm_name2idx']
        self.kpi_name2idx = map_dict['kpi_name2idx']
        self.nftype_name2idx = map_dict['nftype_name2idx']
        self.label_name2idx = map_dict['label_name2idx']

        self.attr_mapping = json.load(open(self.args.important_attr_path, 'r'))

        attr_list = list(json.load(open('./pretrain/fault_attr.json', 'r')).keys())
        self.id2idx = {attr_id: idx for idx, attr_id in enumerate(attr_list)}

        self.g = self.pkg2g(fault_pkg)

        self.fault_type_label = self.label_name2idx[fault_pkg.label.fault_type]
        self.fault_ins_label = []
        for ent in fault_pkg.ent_list:
            if ent['ne_name_ori'] == fault_pkg.label.fault_instance:
                self.fault_ins_label.append(1)
            else:
                self.fault_ins_label.append(-1)

    def pkg2g(self, fault_pkg):
        # src_ids = torch.tensor([tri[0] for tri in fault_pkg.tri_list] +
        #                        [tri[1] for tri in fault_pkg.tri_list])
        # dst_ids = torch.tensor([tri[1] for tri in fault_pkg.tri_list] +
        #                        [tri[0] for tri in fault_pkg.tri_list])
        src_ids = torch.tensor([tri[0] for tri in fault_pkg.tri_list])
        dst_ids = torch.tensor([tri[1] for tri in fault_pkg.tri_list])
        g = dgl.graph((src_ids, dst_ids))

        # nftype
        nftype = torch.zeros(g.num_nodes(), dtype=torch.int64)
        for node_idx, ent in enumerate(fault_pkg.ent_list):
            nftype[node_idx] = self.nftype_name2idx[ent['nf_type']]
        g.ndata['nftype'] = nftype

        # alarm_feat
        alarm_feat = torch.zeros((g.num_nodes(), len(self.alarm_name2idx)))
        for k, v in fault_pkg.ent_alarm.items():
            for a in v:
                alarm_feat[k][self.alarm_name2idx[a.alarm_name]] += 1

        g.ndata['alarm_feat'] = alarm_feat

        alarm_attr_feat = torch.zeros((g.num_nodes(), len(self.id2idx)))
        for ent, alarm in fault_pkg.ent_alarm.items():
            for a in alarm:
                alarm_attr_feat[ent][self.id2idx[a.attr_id]] += 1

        g.ndata['alarm_attr_feat'] = alarm_attr_feat

        # kpi_feat
        kpi_feat = torch.zeros((g.num_nodes(), len(self.kpi_name2idx)))
        for k, v in fault_pkg.ent_kpi.items():
            for a in v:
                kpi_feat[k][self.kpi_name2idx[a.kpi_name]] += 1

        g.ndata['kpi_feat'] = kpi_feat

        kpi_attr_feat = torch.zeros((g.num_nodes(), len(self.id2idx)))
        for ent, kpi in fault_pkg.ent_kpi.items():
            for k in kpi:
                kpi_attr_feat[ent][self.id2idx[k.attr_id]] += 1

        g.ndata['kpi_attr_feat'] = kpi_attr_feat

        g.ndata['alarm_kpi_attr_feat'] = alarm_attr_feat + kpi_attr_feat
        
        g.ndata['alarm_kpi_feat'] = torch.cat([alarm_feat, kpi_feat], dim = 1)

        return g