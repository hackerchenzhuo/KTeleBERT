import os
import json
import re
import tarfile
from collections import defaultdict as ddict
import pdb


alarm_name2idx = dict()
alarm_idx = 0
kpi_name2idx = dict()
kpi_idx = 0
nftype_name2idx = dict()
nftype_idx = 0
label_name2idx = dict()
label_idx = 0

def get_pkg_list(data_path, rule_set):
    pkg_list = []
    i = 0
    for pkg_path in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, pkg_path)):
            pkg = FaultPKG(os.path.join(data_path, pkg_path), rule_set)
            if pkg.has_mdaf and pkg.has_fault_ins and pkg.statistic['num_ent'] != 0 and \
                    pkg.statistic['num_alarm'] != 0 and pkg.statistic['label'] != '':
                pkg_list.append(pkg)

                global label_name2idx
                global label_idx
                if pkg.label.fault_type not in label_name2idx:
                    label_name2idx[pkg.label.fault_type] = label_idx
                    label_idx += 1

    return pkg_list

class RuleSet(object):
    def __init__(self, args):
        # rule_list
        self.rule_list = []
        with open(args.rule_path, 'r') as f:
            for line in f.readlines():
                rule = line.split()
                self.rule_list.append([(rule[3], rule[2]), (rule[1], rule[0])])

        # rule_dict
        self.rule_dict = ddict(list)
        for r in self.rule_list:
            self.rule_dict[r[1]].append(r[0])
        self.rule_dict = dict(self.rule_dict)


class AlarmInstance(object):
    def __init__(self, alarm_dict):
        self.alarm_topo_id = alarm_dict['id']
        self.alarm_id = alarm_dict['properties']['alarmId']
        self.alarm_name = alarm_dict['properties']['alarmName']
        self.occurTime = alarm_dict['properties']['occurUtc']
        global alarm_name2idx
        global alarm_idx
        if self.alarm_name not in alarm_name2idx:
            alarm_name2idx[self.alarm_name] = alarm_idx
            alarm_idx += 1


class KPIInstance(object):
    def __init__(self, kpi_dict):
        self.event_id = kpi_dict['properties']['eventId']
        self.status = kpi_dict['properties']['exceptionStatus']

        content = json.loads(kpi_dict['properties']['content'])
        self.counter_name = content['counterName']
        self.counter_id = content['counterId']
        self.occurTime = content['exceptionTime']

        self.kpi_name = self.counter_name + '_异常' + self.status

        global kpi_name2idx
        global kpi_idx
        if self.kpi_name not in kpi_name2idx:
            kpi_name2idx[self.kpi_name] = kpi_idx
            kpi_idx += 1


class FaultLabel(object):
    def __init__(self, label_dict):
        self.json = label_dict
        self.fault_type = label_dict['faultType']
        self.fault_instance = label_dict['faultInstances'][0]['rootNeName']

        # global label_name2idx
        # global label_idx
        # if self.fault_type not in label_name2idx:
        #     label_name2idx[self.fault_type] = label_idx
        #     label_idx += 1


class FaultPKG(object):
    def __init__(self, pkg_path, rule_set):
        self.pkg_path = pkg_path
        self.pkg_name = pkg_path.split('/')[-1]
        self.rule_set = rule_set

        # entity and triple
        self.ent_list = None    
        self.ent_n5gtopo_id2idx = None
        self.file_name2idx = None
        self.tri_list = None
        # alarm and kpi
        self.ent_alarm = dict()
        self.ent_kpi = dict()

        # read pkg files
        self.has_mdaf = True
        self.read_pkg()

        # label
        self.has_fault_ins = False
        self.label = self.read_label()

        # get statistic
        self.statistic = self.get_statistic()

        # get root ins
        self.root_ins = self.instance_rule()

    def instance_rule(self):
        '''
            tuple(neName, nfType, fault, faultid)
            
            1. rule, MDAF -> 类型的链 (nfType, faultid) -> {start (nfType, faultid): [link_list]}
            2. 根据位置(目前不需要)、时间、实例化规则链 
            3. 将规则链的尾节点找到统计
        '''
        fault_tuple = []
        nfType_fault_list = []
        tuple2instance = {}
        for ent, alarm in self.ent_alarm.items():
            for a in alarm:
                ne_name = self.ent_list[ent]['ne_name']
                label = self.ent_list[ent]['label']
                fault_id = a.alarm_id
                fault_tuple.append((ne_name, label, fault_id, a))
                temp = (label, fault_id)
                if not temp in nfType_fault_list:
                    nfType_fault_list.append(temp)
                    tuple2instance[temp] = [(ne_name, label, fault_id, a)]
                else:
                    tuple2instance[temp].append((ne_name, label, fault_id, a))

        for ent, kpi in self.ent_kpi.items():
            for k in kpi:
                ne_name = self.ent_list[ent]['ne_name']
                label = self.ent_list[ent]['label']
                fault_id = k.counter_id
                fault_tuple.append((ne_name, label, str(fault_id), k))
                temp = (label, str(fault_id) + " _异常" + k.status)
                if not temp in nfType_fault_list:
                    nfType_fault_list.append(temp)
                    tuple2instance[temp] = [(ne_name, label, str(fault_id) + " _异常" + k.status, a)]
                else:
                    tuple2instance[temp].append((ne_name, label, str(fault_id) + " _异常" + k.status, a))

        # fault_tuple = list(set(fault_tuple))
        # fault_tuple_dict = ddict(list)
        # for fault in fault_tuple:
        #     fault_tuple_dict[fault[1:]].append(fault)
        # fault_tuple_dict = dict(fault_tuple_dict)

        '''
             rule, MDAF -> 类型的链 (nfType, faultid) -> {start (nfType, faultid): [link_list]}
        '''
        links = []
        def get_link(tuple, link, rule, instance_list):
            flag = False
            if tuple in rule:
                neighbor_list = rule[tuple]
                for neighbor in neighbor_list:
                    if neighbor in instance_list:
                        if not neighbor in link:
                            flag = True
                            temp = link
                            temp.append(neighbor)
                            get_link(neighbor, temp, rule, instance_list)
            if not flag:
                nonlocal links
                if len(link) > 1:
                    if not link in links:
                        links.append(link)
                return
        for tuple in nfType_fault_list:
            get_link(tuple, [tuple], self.rule_set.rule_dict, nfType_fault_list)

        '''
            将规则实例化
        '''
        inRange = []
        counter = {}
        instance_tuple_list = []
        for domain_tuple in nfType_fault_list:
            for range_tuple in nfType_fault_list:
                if domain_tuple in self.rule_set.rule_dict:
                    if range_tuple in self.rule_set.rule_dict[domain_tuple]:
                        for domain_instance in tuple2instance[domain_tuple]:
                            for range_instance in tuple2instance[range_tuple]:
                                if domain_instance[3].occurTime <= range_instance[3].occurTime:
                                    if not range_instance in inRange:
                                        inRange.append(range_instance)
                                    if not domain_instance in counter:
                                        counter[domain_instance] = 1
                                    else:
                                        counter[domain_instance] += 1
                                    instance_tuple_list.append((domain_instance, range_instance))

        # for instance in counter:
        #     if not instance in inRange:
        #         print(instance, counter[instance])


        # '''
        #     2. 根据位置(目前不需要)、时间、实例化规则链
        # '''
        # if self.pkg_path == "./data/团泊洼实验室数据/20220317_NRF1_NRF故障_RSTCSLBIPPOD":
        #     pdb.set_trace()
        # links_instance = []
        # def get_link_instance(link, index, link_instance, dict):
        #         global tuple2instance
        #         if len(link_instance) == len(link):
        #             nonlocal links_instance
        #             if not link_instance in links_instance:
        #                 links_instance.append(link_instance)
        #             return
        #         else:
        #             candidates = dict[link[index]]
        #             flag = False
        #             for candidate in candidates:
        #                 if len(link_instance) > 0:
        #                     if candidate[3].occurTime > link_instance[-1][3].occurTime:
        #                         temp = link_instance[:index]
        #                         temp.append(candidate)
        #                         flag = True
        #                         get_link_instance(link, index+1, temp, dict)
        #                 else:
        #                     flag = True
        #                     get_link_instance(link, 1, [candidate], dict)
        #             if not flag:
        #                 return
        #         return
        # for link in links:
        #     get_link_instance(link, 0, [], tuple2instance)


        # ins_rule = []
        # for fault1 in fault_tuple:
        #     if fault1[1:] in self.rule_set.rule_dict.keys():
        #         for v in self.rule_set.rule_dict[fault1[1:]]:
        #             if v in fault_tuple_dict:
        #                 for fault2 in fault_tuple_dict[v]:
        #                     ins_rule.append([fault1, fault2])

        # root_ins = list(set([ir[1][0] for ir in ins_rule]))
        # if len(root_ins) > 0:
        #     pdb.set_trace()
        no_root_list = []
        for tuple in inRange:
            if tuple[0] not in no_root_list:
                no_root_list.append(tuple[0])
        root_ins = []
        if not self.ent_list == None:
            for ent in self.ent_list:
                if ent['ne_name'] in no_root_list:
                    root_ins.append(-1)
                else:
                    root_ins.append(1)
        return root_ins

    def read_label(self):
        label_json = json.load(open(os.path.join(self.pkg_path, 'faultLabel.json'), 'r'))
        faultLabel = FaultLabel(label_json)
        if self.has_mdaf and faultLabel.fault_instance in [e['ne_name_ori'] for e in self.ent_list]:
            self.has_fault_ins = True

        return FaultLabel(label_json)

    def get_statistic(self):
        statistic = {
            'path': self.pkg_path,
            'label': self.label.fault_type,
            'has_mdaf': self.has_mdaf,
            'has_fault_ins': self.has_fault_ins,
            'num_ent': len(self.ent_list) if self.has_mdaf else 0,
            'num_tri': len(self.tri_list) if self.has_mdaf else 0,
            'num_alarm': sum([len(a) for a in self.ent_alarm.values()]) if self.has_mdaf else 0,
            'num_kpi': sum([len(k) for k in self.ent_kpi.values()]) if self.has_mdaf else 0,
        }

        return statistic

    def extract(self, tar_path, target_path):
        try:
            tar = tarfile.open(tar_path, "r:gz")
            file_names = tar.getnames()
            for file_name in file_names:
                tar.extract(file_name, target_path)
            tar.close()
        except Exception as e:
            print(e)

    def read_pkg(self):
        mdaf_list = os.listdir(os.path.join(self.pkg_path, 'MDAF'))
        if not len(mdaf_list) == 0:
            if len(mdaf_list) == 1:
                self.extract(os.path.join(self.pkg_path, 'MDAF', mdaf_list[0]),
                             os.path.join(self.pkg_path, 'MDAF'))

            self.ent_list, self.ent_n5gtopo_id2idx, self.file_name2idx = self.read_ent()
            self.tri_list = self.read_tri()
            self.read_fault()
        else:
            self.has_mdaf = False

    def read_ent(self):
        ent_file_path = os.path.join(self.pkg_path, 'MDAF', 'n5gtopo', 'source_entity.json')
        ent_file = json.load(open(ent_file_path, 'r'))
        ent_list = []
        for idx, ent in enumerate(ent_file):
            n5gtopo_id = ent["id"]

            ne_id = int(re.match(r'NE=(\d*)', ent["properties"]["vnfDn"]).group(1))

            nf_type = ent["properties"]["nfType"]
            # if nf_type == "UPF":
            #     nf_type = "UDG"

            ne_type = ent["properties"]["neType"]

            label = ent["label"]

            ne_name = ent["properties"]["neName"]
            if len(ne_name) > 6:
                if len(ne_name.split("HXBXAsn")) == 1:
                    ne_name = ne_name.split("XA")[1]
                else:
                    ne_name = ne_name.split("HXBXAsn")[1]
                ne_name = ne_name.split("BHW")[0]
                _ = ne_name.split("00")
                ne_name = _[0] + _[1]

            ent_dict = {'n5gtopo_id': n5gtopo_id,
                        'ne_id': ne_id,
                        'nf_type': nf_type,
                        'ne_type': ne_type,
                        'ne_name': ne_name,
                        'ne_name_ori': ent["properties"]["neName"],
                        'file_name': f'NE={ne_id}_{ne_type}_{ent["properties"]["neName"]}',
                        'label': label
                        }

            ent_list.append(ent_dict)

            global nftype_name2idx
            global nftype_idx
            if nf_type not in nftype_name2idx:
                nftype_name2idx[nf_type] = nftype_idx
                nftype_idx += 1

        ent_n5gtopo_id2idx = {ent['n5gtopo_id']: idx for idx, ent in enumerate(ent_list)}
        file_name2idx = {ent['file_name']: idx for idx, ent in enumerate(ent_list)}

        return ent_list, ent_n5gtopo_id2idx, file_name2idx

    def read_tri(self):
        tri_file_path = os.path.join(self.pkg_path, 'MDAF', 'n5gtopo', 'source_relation.json')
        tri_file = json.load(open(tri_file_path, 'r'))
        tri_list = []
        for tri in tri_file:
            try:
                tri_list.append((self.ent_n5gtopo_id2idx[tri["outV"]],
                                 self.ent_n5gtopo_id2idx[tri["inv"]],
                                 tri["properties"]["relationType"]))
            except KeyError:
                if tri["outV"] not in self.ent_n5gtopo_id2idx:
                    # pass
                    print(f'{self.pkg_path}: {tri["outV"]} in source_relation.json but not in source_entity.json')
                    # print(f"{self.pkg_path.split('/')[2] + '/' + self.pkg_path.split('/')[3]}, "
                    #       f"{tri['outV']}")
                if tri["inv"] not in self.ent_n5gtopo_id2idx:
                    # pass
                    print(f'{self.pkg_path}: {tri["inv"]} in source_relation.json but not in source_entity.json')
                    # print(f"{self.pkg_path.split('/')[2] + '/' + self.pkg_path.split('/')[3]}, "
                    #       f"{tri['inv']}")

        tri_list = list(set(tri_list))

        return tri_list

    def read_fault(self):
        fault_file_list = os.listdir(os.path.join(self.pkg_path, 'MDAF'))
        for ent_idx, ent in enumerate(self.ent_list):
            if ent['file_name'] in fault_file_list:
                alarm_list = self.read_alarm(ent['file_name'])
                self.ent_alarm[ent_idx] = alarm_list

                kpi_list = self.read_kpi(ent['file_name'])
                self.ent_kpi[ent_idx] = kpi_list

    def read_alarm(self, fault_file_path):
        alarm_list = []
        for alarm_file_path in os.listdir(os.path.join(self.pkg_path, 'MDAF', fault_file_path)):
            if 'alarm' in alarm_file_path:
                alarm_file = open(os.path.join(self.pkg_path, 'MDAF', fault_file_path, alarm_file_path))
                alarm_json = json.load(alarm_file)
                for alarm in alarm_json:
                    alarm_list.append(AlarmInstance(alarm))

        return alarm_list

    def read_kpi(self, fault_file_path):
        kpi_list = []
        for kpi_file_path in os.listdir(os.path.join(self.pkg_path, 'MDAF', fault_file_path)):
            if 'kpi' in kpi_file_path:
                kpi_file = open(os.path.join(self.pkg_path, 'MDAF', fault_file_path, kpi_file_path))
                kpi_json = json.load(kpi_file)
                for kpi in kpi_json:
                    kpi_list.append(KPIInstance(kpi))

        return kpi_list


if __name__ == '__main__':
    data_path = './data/团泊洼实验室数据'
    pkg_list = []
    for pkg_path in os.listdir(data_path):
        pkg = FaultPKG(os.path.join(data_path, pkg_path))
        if pkg.has_mdaf and pkg.statistic['num_ent'] != 0 and \
                pkg.statistic['num_alarm'] != 0 and pkg.statistic['label'] != '':
            pkg_list.append(pkg)

    pass

    # for k, v in kpi_id2name.items():
    #     for k1, v1 in kpi_id2name.items():
    #         if v == v1 and k != k1:
    #         print(k, v, k1, v1)