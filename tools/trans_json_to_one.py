import json

# 输入文件与输出文件
json_load = ["data/track2&3/1.json","data/track2&3/2.json","data/track2&3/3.json",
            "data/track2&3/4.json","data/track2&3/5.json","data/track2&3/6.json",
            "data/track2&3/7.json","data/track2&3/8.json","data/track2&3/9.json",
            "data/track2&3/10.json","data/track2&3/11.json","data/track2&3/12.json",
            "data/track2&3/13.json","data/track2&3/14.json","data/track2&3/15.json",
            "data/track2&3/16.json","data/track2&3/17.json","data/track2&3/18.json",
            "data/track2&3/19.json"]

json_save = "data/track2&3/all.json"
data_save = {"annolist":[]}

print("---Start---")

for i_dir, json_file in enumerate(json_load):
    i_dir = i_dir + 1
    print("---Loading",json_file,"---")
    with open(json_file,'r') as f:
        data = json.load(f)

    print("Load ", len(data['annolist']), "annolist")

    for i in range(len(data['annolist'])):
        if i_dir < 10:
            img_name = "0" + str(i_dir) + '_' + data['annolist'][i]["image"][0]["name"]
        else:
            img_name = str(i_dir) + '_' + data['annolist'][i]["image"][0]["name"]
        data['annolist'][i]["image"][0]["name"] = img_name
        data_save["annolist"].append(data['annolist'][i])
    
    print("Dir", i_dir, "Complete")

print("---Saving---")
with open(json_save,"w") as f:
    json.dump(data_save,f)