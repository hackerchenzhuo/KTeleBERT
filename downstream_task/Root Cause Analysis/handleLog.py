import os
import pdb
import pandas as pd
import argparse

def get_filelist(dir, Filelist, keyword):
    newDir = dir

    if os.path.isfile(dir):
        if keyword in dir:
            Filelist.append(dir)

    elif os.path.isdir(dir):

        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)

            get_filelist(newDir, Filelist, keyword)

    return Filelist

parser = argparse.ArgumentParser()
parser.add_argument('--keyword', default='')

args = parser.parse_args()


allLogList = get_filelist("./log", [], args.keyword)

logList = []

for log in allLogList:
    if "run" not in log:
        logList.append(log)

data = {}

for log in logList:
    name = log[0 : log.find(".log") - 2].split("/")[-1]
    if not name in data:
        data[name] = {}
    result = ""
    with open(log, "r", encoding="utf-8") as f:
        for line in f:
            result = line
    data[name][log[-5]] = {}
    results = result.strip().split("\t")
    data[name][log[-5]]["MR"] = float(results[1])
    data[name][log[-5]]["Hit@1"] = float(results[2])
    data[name][log[-5]]["Hit@3"] = float(results[3])            
    data[name][log[-5]]["Hit@10"] = float(results[4])
    
for name in data:
    if len(data[name]) == 3:
        data[name]["avg"] = {}
        data[name]["avg"]["MR"] = (data[name]["1"]["MR"] + data[name]["2"]["MR"] + data[name]["3"]["MR"]) / 3
        data[name]["avg"]["Hit@1"] = (data[name]["1"]["Hit@1"] + data[name]["2"]["Hit@1"] + data[name]["3"]["Hit@1"]) / 3
        data[name]["avg"]["Hit@3"] = (data[name]["1"]["Hit@3"] + data[name]["2"]["Hit@3"] + data[name]["3"]["Hit@3"]) / 3
        data[name]["avg"]["Hit@10"] = (data[name]["1"]["Hit@10"] + data[name]["2"]["Hit@10"] + data[name]["3"]["Hit@10"]) / 3

import xlwt
workbook = xlwt.Workbook(encoding= 'utf-8')
worksheet_result = workbook.add_sheet("result")

for i, name in enumerate(data):
    if "avg" in data[name]:   
        worksheet_result.write(i, 0, name)
        worksheet_result.write(i, 1, data[name]["avg"]["MR"])
        worksheet_result.write(i, 2, data[name]["avg"]["Hit@1"])
        worksheet_result.write(i, 3, data[name]["avg"]["Hit@3"])
        worksheet_result.write(i, 4, data[name]["avg"]["Hit@10"])

workbook.save("result"+args.keyword+".xls")


