import json
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

model = AutoModel.from_pretrained("klue/bert-base")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

class KlueBERTClassifier(nn.Module):
    def __init__(self, config):
        super(KlueBERTClassifier, self).__init__(config)

        #klue bert
        self.bert = AutoModel.from_pretrained("klue/bert-base")
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = config.hidden_size
        self.max_length = config.max_length
        self.num_labels = config.num_labels
        self.linear = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, ):



inputs = tokenizer('축구는 정말 재미있는 [MASK]다.', return_tensors='pt')
print(inputs['input_ids'])

#
# Printass inputs through KLUE-BERT model
outputs = model(**inputs)

# Extract output of [CLS] token from last layer
last_hidden_state = outputs.last_hidden_state
cls_output = last_hidden_state[:, 0, :]

# Add linear layer on top of [CLS] output
linear_layer = torch.nn.Linear(cls_output.shape[-1], 1)
linear_output = linear_layer(cls_output)

# Print output
print(linear_output)



