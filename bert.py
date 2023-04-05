import os
import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertPreTrainedModel, BertModel
from torch.utils.data import (DataLoader, TensorDataset, RandomSampler)
import torch.optim as optim
import numpy as np
from transformers import AutoModel, AutoTokenizer
from transformers import BertConfig
from tokenization_kobert import KoBertTokenizer

import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class SentimentClassifier(BertPreTrainedModel):

    def __init__(self, config):
        super(SentimentClassifier, self).__init__(config)

        # BERT 모델
        self.bert = AutoModel.from_pretrained("klue/bert-base")

        # 히든 사이즈 768
        self.hidden_size = config.hidden_size

        self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size // 2, dropout=0.3, num_layers=1,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        # 분류할 라벨의 개수 2
        self.num_labels = config.num_labels

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.num_labels)

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)

        # BERT 출력에서 CLS에 대응하는 벡터 표현 추출
        # 선형 함수를 사용하여 예측 확률 분포로 변환
        # (batch_size, max_lenth, hidden_size)
        bert_output = outputs[0]

        bert_output = self.dropout(bert_output)

        # outputs : (batch,length,2*hidden/2)
        # hidden_output : (2,batch, hidden/2)
        outputs, hidden_output = self.rnn(bert_output)

        # print(len(outputs))
        # (batch_size, hidden_size)
        batch_size, max_length, _ = outputs.shape

        # (batch*length, hidden)
        re_outputs = outputs.reshape(batch_size * max_length, -1)

        linear_output = self.linear(re_outputs)
        linear_output = linear_output.reshape(batch_size, max_length, -1)
        # class_output : (batch_size, num_labels)
        cls_output = linear_output[:, 0, :]

        return cls_output


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
    return datas


def read_vocab_data(vocab_data_path):
    term2idx, idx2term = {}, {}  # {"<PAD>": 0}, {0: "<PAD>"}

    with open(vocab_data_path, "r", encoding="utf8") as inFile:
        lines = inFile.readlines()

    for line in lines:
        term = line.strip()
        term2idx[term] = len(term2idx)
        idx2term[term2idx[term]] = term

    # print(term2idx)
    # print(idx2term)
    return term2idx, idx2term


def convert_data2feature(datas, max_length, tokenizer, label2idx):
    input_ids_features, label_id_features = [], []

    for input_sequence, label in datas:
        # CLS, SEP 토큰 추가
        tokens = [tokenizer.cls_token]
        tokens += input_sequence
        tokens = tokens[:max_length - 1]
        tokens += [tokenizer.sep_token]

        # word piece들을 대응하는 index로 치환
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # padding 생성
        padding = [tokenizer._convert_token_to_id(tokenizer.pad_token)] * (max_length - len(input_ids))
        input_ids += padding

        label_id = label2idx[label]

        # 변환한 데이터를 각 리스트에 저장
        input_ids_features.append(input_ids)
        label_id_features.append(label_id)

    # 변환한 데이터를 Tensor 객체에 담아 반환
    input_ids_features = torch.tensor(input_ids_features, dtype=torch.long)
    label_id_features = torch.tensor(label_id_features, dtype=torch.long)

    return input_ids_features, label_id_features


def train(config):
    # BERT config 객체 생성
    bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
                                             cache_dir=config["cache_dir_path"])
    setattr(bert_config, "num_labels", config["num_labels"])

    # BERT tokenizer 객체 생성
    bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    # 라벨 딕셔너리 생성
    label2idx, idx2label = read_vocab_data(vocab_data_path=config["label_vocab_data_path"])

    # 학습 및 평가 데이터 읽기
    train_datas = read_data(file_path=config["train_data_path"])

    # 입력 데이터 전처리
    train_input_ids_features, train_label_id_features = convert_data2feature(datas=train_datas,
                                                                             max_length=config["max_length"],
                                                                             tokenizer=bert_tokenizer,
                                                                             label2idx=label2idx)

    # 학습 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    train_dataset = TensorDataset(train_input_ids_features, train_label_id_features)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"],
                                  sampler=RandomSampler(train_dataset))

    # 사전 학습된 BERT 모델 파일로부터 가중치 불러옴
    model = SentimentClassifier.from_pretrained(pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
                                                cache_dir=config["cache_dir_path"], config=bert_config).cuda()

    # loss를 계산하기 위한 함수
    loss_func = nn.CrossEntropyLoss()

    # 모델 학습을 위한 optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    for epoch in range(config["epoch"]):
        model.train()

        total_loss = []
        for batch in train_dataloader:
            batch = tuple(t.cuda() for t in batch)
            input_ids, label_id = batch

            # 역전파 단계를 실행하기 전에 변화도를 0으로 변경
            optimizer.zero_grad()

            # hypothesis : [batch, num_labels]
            # 모델 예측 결과

            hypothesis = model(input_ids)

            # loss 계산
            loss = loss_func(hypothesis, label_id)

            # loss 값으로부터 모델 내부 각 매개변수에 대하여 gradient 계산
            loss.backward()
            # 모델 내부 각 매개변수 가중치 갱신
            optimizer.step()

            # batch 단위 loss 값 저장
            total_loss.append(loss.data.item())

        bert_config.save_pretrained(save_directory=config["output_dir_path"])
        model.save_pretrained(save_directory=config["output_dir_path"])

        print("Average loss : {}".format(np.mean(total_loss)))


def test(config):
    # BERT config 객체 생성
    bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=config["output_dir_path"],
                                             cache_dir=config["cache_dir_path"])

    # BERT tokenizer 객체 생성 (기존 BERT tokenizer 그대로 사용)
    bert_tokenizer = KoBertTokenizer.from_pretrained(
        pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
        cache_dir=config["cache_dir_path"])

    # 라벨 딕셔너리 생성
    label2idx, idx2label = read_vocab_data(vocab_data_path=config["label_vocab_data_path"])

    # 평가 데이터 읽기
    test_datas = read_data(file_path=config["test_data_path"])
    test_datas = test_datas[:100]

    # 입력 데이터 전처리
    test_input_ids_features, test_label_id_features = convert_data2feature(datas=test_datas,
                                                                           max_length=config["max_length"],
                                                                           tokenizer=bert_tokenizer,
                                                                           label2idx=label2idx)

    # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    test_dataset = TensorDataset(test_input_ids_features, test_label_id_features)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"],
                                 sampler=SequentialSampler(test_dataset))

    # 학습한 모델 파일로부터 가중치 불러옴
    model = SentimentClassifier.from_pretrained(pretrained_model_name_or_path=config["output_dir_path"],
                                                cache_dir=config["cache_dir_path"], config=bert_config).cuda()

    model.eval()
    score = 0
    all = 0
    for batch in test_dataloader:
        batch = tuple(t.cuda() for t in batch)
        input_ids, label_id = batch

        with torch.no_grad():
            # 모델 예측 결과
            hypothesis = model(input_ids)
            # 모델의 출력값에 softmax와 argmax 함수를 적용
            hypothesis = torch.argmax(torch.softmax(hypothesis, dim=-1), dim=-1)

        # Tensor를 리스트로 변경
        hypothesis = hypothesis.cpu().detach().numpy().tolist()
        label_id = label_id.cpu().detach().numpy().tolist()

        for index in range(len(input_ids)):
            input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[index])
            input_sequence = bert_tokenizer.convert_tokens_to_string(
                input_tokens[1:input_tokens.index(bert_tokenizer.sep_token)])
            predict = idx2label[hypothesis[index]]
            correct = idx2label[label_id[index]]
            all = all + 1
            if (predict == correct):
                score = score + 1

            print("입력 : {}".format(input_sequence))
            print("출력 : {}, 정답 : {}\n".format(predict, correct))

    print(score)


if (__name__ == "__main__"):
    output_dir = os.path.join("output")
    cache_dir = os.path.join("cache")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    config = {"mode": "train",
              "train_data_path": os.path.join("test.json"),
              "test_data_path": os.path.join("Validation/행사.json"),
              "output_dir_path": output_dir,
              "cache_dir_path": cache_dir,
              "pretrained_model_name_or_path": "klue/bert-base",
              "label_vocab_data_path": os.path.join("label_vocab.txt"),
              "num_labels": 9,
              "max_length": 512,
              "epoch": 10,
              "batch_size": 64
              }

    if (config["mode"] == "train"):
        train(config)
    else:
        test(config)
