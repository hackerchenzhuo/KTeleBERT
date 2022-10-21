import json
from torch.utils.data import Dataset

class MDAFDataset(Dataset):
    def __init__(self, mode='train', use_pretrain=False):
        data_file = json.load(open("./data/{}.json".format(mode), "r"))
        self.scene_name_map = json.load(open("./data/scene.json".format(mode), "r"))
        self.text_map = json.load(open("./data/event2text.json", 'r'))
        self.time_table = json.load(open("./data/time.json", 'r'))
        self.text_data = []
        self.not_in = []
        for single_data in data_file:
            ent1, ent2 = single_data["ent1"], single_data["ent2"]
            if True:
                if len(ent1) > 10:
                    ent1 = ent1[5:]
                if len(ent2) > 10:
                    ent2 = ent2[5:]
            scene = single_data["scene"]
            label = single_data["label"]
            if ent1 in self.text_map.keys() and ent2 in self.text_map.keys():
                self.text_data.append({
                    "ent1": ent1, 
                    "ent2": ent2, 
                    "scene": scene, 
                    "label": label, 
                    "scene_text": self.get_scene_name(scene),
                    "time": self.get_time_diff(ent1, ent2, scene)
                })
    
    def get_scene_name(self, scene):
        for i in self.scene_name_map.keys():
            if i in scene:
                return i
        return ""
    

    def get_time_diff(self, ent1, ent2, scene):
        scene_time = self.time_table[scene]
        if ent1 not in scene_time.keys() or ent2 not in scene_time.keys():
            return 0
        else:
            return scene_time[ent2] - scene_time[ent1]


    def __len__(self):
        return len(self.text_data)
    

    def __getitem__(self, key):
        return self.text_data[key]
