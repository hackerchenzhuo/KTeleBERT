import os
import json

scene_data_path = "./data/processed-new/mdaf-new/"

def load_scene_data():
    scene_files = os.listdir(scene_data_path)
    res = []
    node_map = {}
    scene_graphs = {}
    for scene_file in scene_files:
        scene_path = os.path.join(scene_data_path, scene_file)
        scene_data = json.load(open(scene_path, 'r'))["graph"]
        nodes = scene_data["nodes"]
        links = scene_data["links"]
        scene_graphs[scene_file] = []
        for node in nodes:
            if node not in node_map.keys():
                node_map[node] = len(node_map)
            scene_graphs[scene_file].append(node_map[node])
    for i in scene_graphs.keys():
        if scene_graphs[i] == []:
            scene_graphs[i] = [len(node_map)]
    return node_map, scene_graphs


def load_tokenizer():
    tokenizer = json.load(open("./data/tokenized.json", 'r'))
    return tokenizer


if __name__ == "__main__":
    a, b = load_scene_data()
    vocab = json.load(open("./data/vocab.json", 'r'))
    inv_vocab = {}
    scene_name_map = {}
    for k in vocab.keys():
        inv_vocab[vocab[k]] = k
    for key in b.keys():
        tokens = b[key]
        scene_name = "".join(inv_vocab[token] for token in tokens)
        scene_name_map[key] = scene_name
    
