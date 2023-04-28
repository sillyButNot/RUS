import json
# Load the JSON data
with open('combined_data_test_100.json', 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# Initialize a counter variable
count = 0

# Iterate through the data array
for item in data['data']:
    if 'topic' in item :
        count = count+1

# Print the count
print(count)


