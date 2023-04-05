import json


def read_data(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    datas = []

    for item in data['data']:
        if 'header' in item and 'dialogueInfo' in item['header']:
            dialogue_info = item['header']['dialogueInfo']
            if ('topic' in dialogue_info):
                label = dialogue_info['topic']
        sentences = []
        if 'body' in item:
            body_ = item['body']
            for i in body_:
                sentences.append(i['utterance'])

        datas.append((sentences, label))
    # Print the count
    print(datas)


read_data("test.json")
