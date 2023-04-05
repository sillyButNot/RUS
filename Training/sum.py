import json

filenames = ["개인및관계.json","미용과건강.json", "상거래(쇼핑).json", "시사교육.json", "식음료.json",
             "여가생활.json", "일과직업.json", "주거와생활.json", "행사.json"]

combined_data = {"data": []}

for filename in filenames:
    with open(filename, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    data = json_data['data']

    combined_data['data'].extend(data)
    with open('combined_data6.json', 'w', encoding='utf-8-sig') as outfile:
        outfile.write(json.dumps(combined_data, indent=4, sort_keys=False, ensure_ascii=False))
    # if 'data' in json_data:
    #     body_list = json_data['data']
    #     combined_data['data'].extend(body_list)
    #     print("if")
    #     print(filename)
    # else:
    #     print("nn")
    #     print(filename)
    # with open('combined_data1.json', 'w', encoding='utf-8-sig') as outfile:
    #     json_string = json.dumps(combined_data, indent=4, sort_keys=True, ensure_ascii=False)
    #     outfile.write(json_string)




