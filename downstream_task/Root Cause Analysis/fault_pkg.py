import os
import json
import re
import tarfile
from collections import defaultdict as ddict
from utils import StreamLog
import torch

pkg_proc_log = StreamLog('Process Data').get_logger()


def get_pkg_list(args, data_path, rule_set):
    pkg_list = []
    for pkg_path in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, pkg_path)):
            pkg = FaultPKG(args, os.path.join(data_path, pkg_path), rule_set)
            if pkg.has_mdaf and pkg.has_fault_ins and pkg.statistic['num_ent'] != 0 and \
                    pkg.statistic['num_alarm'] != 0 and pkg.statistic['label'] != '' and pkg.label_has_fault:
                pkg_list.append(pkg)

    return pkg_list


class RuleSet(object):
    def __init__(self, rule_path):
        # rule_list
        self.rule_list = []
        with open(rule_path, 'r') as f:
            for line in f.readlines():
                rule = line.split()
                self.rule_list.append([(rule[1], rule[0]), (rule[3], rule[2])])

        # rule_dict
        self.rule_dict = ddict(list)
        for r in self.rule_list:
            self.rule_dict[r[0]].append(r[1])
        self.rule_dict = dict(self.rule_dict)


class AlarmInstance(object):
    def __init__(self, alarm_dict, attr_mapping, nf_type):
        self.alarm_dict = alarm_dict
        self.alarm_topo_id = alarm_dict['id']
        self.alarm_id = alarm_dict['properties']['alarmId']
        self.alarm_name = alarm_dict['properties']['alarmName']
        self.occurTime = alarm_dict['properties']['occurUtc']

        severity = attr_mapping['severity_mappings'][alarm_dict['properties']['severity']]
        self.attr_id = f"{self.alarm_name}&&{nf_type}&&{severity}"


class KPIInstance(object):
    def __init__(self, kpi_dict, nf_type):
        self.kpi_dict = kpi_dict
        self.event_id = kpi_dict['properties']['eventId']
        self.status = kpi_dict['properties']['exceptionStatus']

        self.content = json.loads(kpi_dict['properties']['content'])
        self.counter_name = self.content['counterName']
        self.counter_id = str(self.content['counterId'])
        self.occurTime = self.content['exceptionTime']

        self.kpi_id = self.counter_id + f'_异常{self.status}'
        self.kpi_name = self.counter_name + ' 异常' + self.status  # 在rule里面使用

        self.attr_id = f"{self.counter_name}&&{self.status}&&{nf_type}"


class FaultLabel(object):
    def __init__(self, label_dict):
        self.json = label_dict
        self.fault_type = label_dict['faultType']
        self.fault_instance = label_dict['faultInstances'][0]['rootNeName']


class FaultPKG(object):
    def __init__(self, args, pkg_path, rule_set):
        self.args = args
        self.attr_mapping = json.load(open(args.important_attr_path, 'r'))

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
        self.label_has_fault = True

        # get statistic
        self.statistic = self.get_statistic()

        # get instance rule
        if self.has_mdaf and rule_set is not None:
            self.ins_rule_list = self.instance_rule()
            self.process_ins_rule()
            self.fault_tuple = []

    def instance_rule(self):
        '''
            tuple(neName, nfType, fault, faultid)

            1. rule, MDAF -> 类型的链 (nfType, faultid) -> {start (nfType, faultid): [link_list]}
            2. 根据位置(目前不需要)、时间、实例化规则链
            3. 将规则链的尾节点找到统计
        '''
        fault_type_list = []
        fault_tuple = []
        type2ins = {}
        for ent, alarm in self.ent_alarm.items():
            for a in alarm:
                ne_name = self.ent_list[ent]['ne_name']
                label = self.ent_list[ent]['label']
                fault_id = a.alarm_id
                curr_fault_ins = (ne_name, label, fault_id, a)
                fault_tuple.append(curr_fault_ins)
                curr_fault_type = (label, fault_id)
                if curr_fault_type not in fault_type_list:
                    fault_type_list.append(curr_fault_type)
                    type2ins[curr_fault_type] = [curr_fault_ins]
                else:
                    type2ins[curr_fault_type].append(curr_fault_ins)

        for ent, kpi in self.ent_kpi.items():
            for k in kpi:
                ne_name = self.ent_list[ent]['ne_name']
                label = self.ent_list[ent]['label']
                fault_id = k.kpi_id
                curr_fault_ins = (ne_name, label, fault_id, k)
                fault_tuple.append(curr_fault_ins)
                curr_fault_type = (label, fault_id)
                if curr_fault_type not in fault_type_list:
                    fault_type_list.append(curr_fault_type)
                    type2ins[curr_fault_type] = [curr_fault_ins]
                else:
                    type2ins[curr_fault_type].append(curr_fault_ins)

        '''
            将规则实例化
        '''
        self.fault_tuple = fault_tuple
        ins_rule_list = []
        for domain_tuple in fault_type_list:
            if domain_tuple in self.rule_set.rule_dict:
                for range_tuple in fault_type_list:
                    if range_tuple in self.rule_set.rule_dict[domain_tuple]:
                        for domain_instance in type2ins[domain_tuple]:
                            for range_instance in type2ins[range_tuple]:
                                # if domain_instance[3].occurTime <= range_instance[3].occurTime:
                                ins_rule_list.append((domain_instance, range_instance))

        return ins_rule_list

    def process_ins_rule(self):
        inRange = []
        counter = {}
        for ins in self.ins_rule_list:
            if not ins[1] in inRange:
                inRange.append(ins[1])
                if not ins[0] in counter:
                    counter[ins[0]] = 1
                else:
                    counter[ins[0]] += 1
        root_list = []
        for tuple in counter.keys():
            if tuple not in inRange:
                root_list.append(tuple)
        if self.ent_list != None:
            root_ins = [1] * len(self.ent_list)
            ne_ent = {}
            for i, ent in enumerate(self.ent_list):
                ne_ent[ent["ne_name"]] = i

            alarm_counted  = []

            for fault in self.fault_tuple:
                if fault not in inRange:
                    # if fault[2] not in alarm_counted:
                    root_ins[ne_ent[fault[0]]] += 0.1   
                    alarm_counted.append(fault[2])
                # if root_ins[ne_ent[fault[0]]]  > 800:
                #     pdb.set_trace()
            if self.args.use_rule == "mask":
                for i in range(len(root_ins)):
                    if root_ins[i] == 1:
                        root_ins[i] =0
                    else:
                        root_ins[i] =1
            if self.args.use_rule == "no":
                root_ins = [1] * len(self.ent_list)

        else:
            root_ins = []
        self.rule_score = root_ins


    def read_label(self):
        label_json = json.load(open(os.path.join(self.pkg_path, 'faultLabel.json'), 'r'))
        faultLabel = FaultLabel(label_json)
        if self.has_mdaf and faultLabel.fault_instance in [e['ne_name_ori'] for e in self.ent_list]:
            self.has_fault_ins = True

        return FaultLabel(label_json)

    def get_statistic(self):
        # if self.has_fault_ins and self.has_mdaf and len(self.ent_list) != 0:
        #     ne_ent = {}
        #     for i, ent in enumerate(self.ent_list):
        #         ne_ent[ent["ne_name"]] = i
        #     fault = self.label.fault_instance

        #     if len(fault) > 6:
        #         if len(fault.split("HXBXAsn")) == 1:
        #             fault = fault.split("XA")[1]
        #         else:
        #             fault = fault.split("HXBXAsn")[1]
        #         fault = fault.split("BHW")[0]
        #         _ = fault.split("00")
        #         fault = _[0] + _[1]

        #     if len(self.ent_alarm[ne_ent[fault]]) == 0 and len(self.ent_kpi[ne_ent[fault]]) == 0:
        #         num = 0
        #         for alarm in self.ent_alarm:
        #             num += len(self.ent_alarm[alarm])
        #         for kpi in self.ent_kpi:
        #             num += len(self.ent_kpi[kpi])
        #         if num > 0:
        #             pkg_proc_log.debug(f'{self.pkg_path}: label has no fault')
        #             self.label_has_fault = False
        # else:
        #     self.label_has_fault = False

        statistic = {
            'path': self.pkg_path,
            'label': self.label.fault_type,
            'has_mdaf': self.has_mdaf,
            'has_fault_ins': self.has_fault_ins,
            'num_ent': len(self.ent_list) if self.has_mdaf else 0,
            'num_tri': len(self.tri_list) if self.has_mdaf else 0,
            'num_alarm': sum([len(a) for a in self.ent_alarm.values()]) if self.has_mdaf else 0,
            'num_kpi': sum([len(k) for k in self.ent_kpi.values()]) if self.has_mdaf else 0,
            'label_has_fault': self.label_has_fault
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

            self.ent_list, self.ent_n5gtopo_id2idx, \
                self.file_name2idx, self.ne_name2idx = self.read_ent()
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
            if nf_type == 'AUSF':
                continue
            # if nf_type == "UPF":
            #     nf_type = "UDG"

            ne_type = ent["properties"]["neType"]

            ne_name = ent["properties"]["neName"]
            if len(ne_name) > 6:
                if len(ne_name.split("HXBXAsn")) == 1:
                    ne_name = ne_name.split("XA")[1]
                else:
                    ne_name = ne_name.split("HXBXAsn")[1]
                ne_name = ne_name.split("BHW")[0]
                _ = ne_name.split("00")
                ne_name = _[0] + _[1]

            label = ent["label"]

            ent_dict = {
                'n5gtopo_id': n5gtopo_id,
                'ne_id': ne_id,
                'nf_type': nf_type,
                'ne_type': ne_type,
                'ne_name': ne_name,
                'ne_name_ori': ent["properties"]["neName"],
                'file_name': f'NE={ne_id}_{ne_type}_{ent["properties"]["neName"]}',
                'label': label
            }

            ent_list.append(ent_dict)

        ent_n5gtopo_id2idx = {ent['n5gtopo_id']: idx for idx, ent in enumerate(ent_list)}
        file_name2idx = {ent['file_name']: idx for idx, ent in enumerate(ent_list)}
        ne_name2idx = {ent['ne_name']: idx for idx, ent in enumerate(ent_list)}

        return ent_list, ent_n5gtopo_id2idx, file_name2idx, ne_name2idx

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
                    pkg_proc_log.debug(f'{self.pkg_path}: {tri["outV"]} in source_relation.json but not in source_entity.json')
                if tri["inv"] not in self.ent_n5gtopo_id2idx:
                    pkg_proc_log.debug(f'{self.pkg_path}: {tri["inv"]} in source_relation.json but not in source_entity.json')

        tri_list = list(set(tri_list))

        return tri_list

    def read_fault(self):
        fault_file_list = os.listdir(os.path.join(self.pkg_path, 'MDAF'))
        for ent_idx, ent in enumerate(self.ent_list):
            if ent['file_name'] in fault_file_list:
                alarm_list = self.read_alarm(ent['file_name'], ent['nf_type'])
                self.ent_alarm[ent_idx] = alarm_list

                kpi_list = self.read_kpi(ent['file_name'], ent['nf_type'])
                self.ent_kpi[ent_idx] = kpi_list

    def read_alarm(self, fault_file_path, nf_type):
        alarm_list = []
        for alarm_file_path in os.listdir(os.path.join(self.pkg_path, 'MDAF', fault_file_path)):
            if 'alarm' in alarm_file_path:
                alarm_file = open(os.path.join(self.pkg_path, 'MDAF', fault_file_path, alarm_file_path))
                alarm_json = json.load(alarm_file)
                for alarm in alarm_json:
                    alarm_list.append(AlarmInstance(alarm, self.attr_mapping, nf_type))

        return alarm_list

    def read_kpi(self, fault_file_path, nf_type):
        kpi_list = []
        for kpi_file_path in os.listdir(os.path.join(self.pkg_path, 'MDAF', fault_file_path)):
            if 'kpi' in kpi_file_path:
                kpi_file = open(os.path.join(self.pkg_path, 'MDAF', fault_file_path, kpi_file_path))
                kpi_json = json.load(kpi_file)
                for kpi in kpi_json:
                    kpi_list.append(KPIInstance(kpi, nf_type))

        return kpi_list