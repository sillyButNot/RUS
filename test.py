import json


def read_data(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    datas = []

    for item in data['data']:
        if 'topic' in item:
            topic = item['topic']

        sentences = []
        if 'utterance' in item:
            sentence = item['utterance']
            for i in sentence:
                sentences.append(i['utterance'])


        #     if ('topic' in dialogue_info):
        #         label = dialogue_info['topic']
        # sentences = []
        # if 'body' in item:
        #     body_ = item['body']
        #     for i in body_:
        #         sentences.append(i['utterance'])
        #
        # datas.append((sentences, label))
        #


read_data("test.json")
