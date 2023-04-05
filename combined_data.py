import json

filenames = ["개인및관계.json","미용과건강.json", "상거래(쇼핑).json", "시사교육.json", "식음료.json",
             "여가생활.json", "일과직업.json", "주거와생활.json", "행사.json"]

combined_data = {"data": []}
final = {"data": []}
for filename in filenames:
    with open(filename, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    result = {"topic": '', "participantID": [], "utterance": []}
    index = 0
    for item in json_data['data']:
        result['topic'] = item['header']['dialogueInfo']['topic']
        body_ = item['body']
        sentences = []
        participant = []
        for i in body_:
            result['utterance'].append(i['utterance'])
            result['participantID'].append(i['participantID'])
        final['data'].append(result)
        result = {"topic": '', "participantID": [], "utterance": []}

with open('combined_data_test.json', 'w', encoding='utf-8-sig') as outfile:
    outfile.write(json.dumps(final, indent=4, sort_keys=False, ensure_ascii=False))
