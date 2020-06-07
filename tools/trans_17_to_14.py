import json

# 输入文件与输出文件
json_file_17 = "11_result_17.json"
json_file_14 = "result_14.json"

# 加载数据
with open(json_file_17,'r') as f:
    output = json.load(f)

annolist = output['annolist']


for i,annolist_temp in enumerate(annolist):
    # 对每张图片进行循环

    # 加载每个人的数据
    annorect = annolist_temp['annorect']
    for j,annorect_temp in enumerate(annorect):
        # 对每个人进行循环

        # 加载关节点数据
        point = annorect_temp['annopoints'][0]['point']

        point_14 = []
        # 将17改为14
        point_14.append(point[0])
        point_14.append({
            "id":[1],
            "x":[(point[5]["x"][0] + point[6]["x"][0]) / 2],
            "y":[(point[5]["y"][0] + point[6]["y"][0]) / 2],
            "score":[(point[5]["score"][0] + point[6]["score"][0]) / 2]})
        point_14.append(point[6])
        point_14.append(point[8])
        point_14.append(point[10])
        point_14.append(point[5])
        point_14.append(point[7])
        point_14.append(point[9])
        point_14.append(point[12])
        point_14.append(point[14])
        point_14.append(point[16])
        point_14.append(point[11])
        point_14.append(point[13])
        point_14.append(point[15])

        output['annolist'][i]["annorect"][j]["annopoints"][0]['point'] = point_14

# 存储14输出
with open(json_file_14,"w") as f:
    json.dump(output,f)

"""
COCO:   0:鼻子  1:左眼  2:右眼  3:左耳  4:右耳  5:左肩  6:右肩  7:左肘  8:右肘  9:左腕  10:右腕 11:左臀 12:右臀 13:左膝 14:右膝 15:左踝 16:右踝
HIE:    0:鼻子  1:胸    2:右肩  3:右肘  4:右腕  5:左肩  6:左肘  7:左腕  8:右臀  9:右膝  10:右踝 11:左臀 12:左膝 13:左踝
"""