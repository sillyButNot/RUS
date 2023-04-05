import json


def read_data(file_path):
    with open(file_path, "r", encoding='utf-8-sig') as f:
        data = json.load(f)
    datas = []

    for item in data['data']:
        if 'topic' in item:
            topic = item['topic']

        sentences = []
        if 'utterance' in item:
            sentence = item['utterance']
        datas.append((sentence, topic))
    return datas


read_data("combined_data.json")
