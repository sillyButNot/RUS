import json
# Load the JSON data
with open('test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize a counter variable
count = 0

# Iterate through the data array
for item in data['data']:
    if 'header' in item and 'dialogueInfo' in item['header']:
        dialogue_info = item['header']['dialogueInfo']
        if ('topic' in dialogue_info):
            print(dialogue_info['topic'])
            # Increment the counter variable
            count += 1

    body_ = item['body']
    for i in body_:
        print(i['utterance'])
    print()
# Print the count
print(count)


