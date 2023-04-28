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
from torch.utils.data import SequentialSampler
from tqdm import tqdm

import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class SentimentClassifier(BertPreTrainedModel):

    def __init__(self, config):
        super(SentimentClassifier, self).__init__(config)

        # BERT 모델
        self.bert = BertModel(config)
        self.max_length = 512
        # 히든 사이즈
        self.hidden_size = config.hidden_size

        # 분류할 라벨의 개수
        self.num_labels = config.num_labels

        # 주제를 랜덤으로 라벨임베딩해줌
        self.label_embedding_matrix = nn.Parameter(
            torch.randn(size=(self.num_labels, self.hidden_size,), dtype=torch.float32, requires_grad=True))

        self.first_multi_head_attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=8,
                                                                batch_first=True)

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.num_labels)
        self.log_softmax = nn.functional.log_softmax

    def forward(self, input_ids):
        batch_size = input_ids.size()[0]

        outputs = self.bert(input_ids=input_ids)

        # BERT 출력에서 CLS에 대응하는 벡터 표현 추출
        # 선형 함수를 사용하여 예측 확률 분포로 변환

        # (batch_size, max_lenth, hidden_size)
        bert_output = outputs[0]

        # (num_labels, hidden) -> (1. num_labels, hidden)
        label_embedding = self.label_embedding_matrix.clone().unsqueeze(dim=0)

        # (1,number_labels, hidden) -> (batch, num_labels, hidden)
        label_embedding = label_embedding.repeat(batch_size, 1, 1)

        # 패딩 마스크
        # (batch_size, max_lenth, hidden_size) -> (batch_size, max_length)

        padding_mask = (input_ids != 0).float()
        padding_mask = padding_mask.unsqueeze(dim=-1)

        padding_mask = padding_mask.repeat(1, 1, self.num_labels)

        # (batch, max_length, hidden)
        # first_attention_outputs : (batch, max_length, hidden)
        # first_attention_weights : (batch, max_length, num_labels)
        first_attention_outputs, first_attention_weights = self.first_multi_head_attention(
            query=bert_output,
            key=label_embedding,
            value=label_embedding)

        #상위 3개 값만 놔두고 나머지 확률값은 0으로 만듬
        topk_values, topk_indices = torch.topk(first_attention_weights, k=3, dim=-1)
        first_attention_weights.zero_()
        first_attention_weights.scatter_(-1, topk_indices, topk_values)


        # # (batch, max_length, hidden) -> (batch, max_length, num_labels)
        # linear_output = self.linear(first_attention_outputs)
        #
        # # (batch, max_length, num_labels)
        # probs = self.log_softmax(linear_output, dim=-1)

        # (batch_size, max_length) -> (batch_size, max_length, labels)
        # padding_mask = padding_mask.repeat(1, 1,self.num_labels)
        probs = first_attention_weights * padding_mask

        # 안되면 sum 말고 average
        # (batch, current_length, num_labels) -> (batch, num_labels)
        topics = probs.sum(dim=1)
        max_length = int(padding_mask.sum(dim=1).max().item())
        topics = topics / max_length

        return topics

def read_data(file_path, bert_tokenizer):
    with open(file_path, "r", encoding='utf-8-sig') as f:
        data = json.load(f)
    datas = []
    person_data = []
    for item in tqdm(data['data'], desc='read_data'):
        if 'topic' in item:
            topic = item['topic']

        sentences = []
        if 'utterance' in item:
            sentence = item['utterance']
            sentence = bert_tokenizer(sentence)['input_ids']
            re_sentence = [inner_sentence[1:-1] for inner_sentence in sentence]

            datas.append((re_sentence, topic))
        personal = []
        if 'participantID' in item:
            personal = item['participantID']
        person_data.append(personal)

    return datas, person_data


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


def convert_data2feature(datas, max_length, tokenizer, label2idx, personal):
    input_ids_features, label_id_features = [], []
    x = 0
    for input_sequence, label in tqdm(datas, desc='convert_data2feature'):
        # CLS, SEP 토큰 추가
        y = 0
        tokens = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])
        for sentence in input_sequence:
            tokens.extend(sentence)
            if y is not (len(personal[x]) - 1):
                if personal[x][y] != personal[x][y + 1]:
                    tokens = tokens[:max_length - 1]
                    tokens += tokenizer.convert_tokens_to_ids([tokenizer.sep_token])
            else:
                tokens += tokenizer.convert_tokens_to_ids([tokenizer.sep_token])

            y = y + 1
        x = x + 1
        # word piece들을 대응하는 index로 치환
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokens
        # padding 생성
        if len(input_ids) > 512:
            input_ids = input_ids[:512]
        padding = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (max_length - len(input_ids))

        input_ids += padding
        if len(input_ids) > 512:
            print()

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
    bert_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
        cache_dir=config["cache_dir_path"])

    # 라벨 딕셔너리 생성
    label2idx, idx2label = read_vocab_data(vocab_data_path=config["label_vocab_data_path"])

    # 학습 및 평가 데이터 읽기
    train_datas, personal = read_data(file_path=config["train_data_path"], bert_tokenizer=bert_tokenizer)

    # 입력 데이터 전처리
    train_input_ids_features, train_label_id_features = convert_data2feature(datas=train_datas,
                                                                             max_length=config["max_length"],
                                                                             tokenizer=bert_tokenizer,
                                                                             label2idx=label2idx, personal=personal)

    # 학습 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    train_dataset = TensorDataset(train_input_ids_features, train_label_id_features)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"],
                                  sampler=RandomSampler(train_dataset))

    # 사전 학습된 BERT 모델 파일로부터 가중치 불러옴
    model = SentimentClassifier.from_pretrained(pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
                                                cache_dir=config["cache_dir_path"], config=bert_config).cuda()

    # loss를 계산하기 위한 함수
    loss_func = nn.CrossEntropyLoss() #softmax를 거지치지 않은 것이기 때문?

    # 모델 학습을 위한 optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    start_epoch = 0

    if config["checkpoint_path"] is not None:
        checkpoint = torch.load(config["checkpoint_path"], map_location=torch.device('cuda'))

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # 시작 epoch를 이전 상태에서 +1로 설정

        print("Loaded checkpoint from: {}".format(config["checkpoint_path"]))
        print("Resuming from epoch {}".format(start_epoch))

    for epoch in range(start_epoch, config["epoch"]):
        model.train()

        total_loss = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, label_id = batch

            # 역전파 단계를 실행하기 전에 변화도를 0으로 변경
            optimizer.zero_grad()

            # hypothesis : [batch, num_labels]
            # 모델 예측 결과

            # (batch, num_labels)
            hypothesis = model(input_ids)

            # loss 계산
            loss = loss_func(hypothesis, label_id)

            # loss 값으로부터 모델 내부 각 매개변수에 대하여 gradient 계산
            loss.backward()
            # 모델 내부 각 매개변수 가중치 갱신
            optimizer.step()

            # batch 단위 loss 값 저장
            total_loss.append(loss.data.item())

            print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(epoch + 1, config["epoch"], step + 1,
                                                                      len(train_dataloader), loss.item()))


        save_dir = os.path.join(config["output_dir_path"], "epoch-{}".format(epoch+1))
        bert_config.save_pretrained(save_directory=save_dir)
        model.save_pretrained(save_directory=save_dir)
        save_checkpoint(save_dir=save_dir, model=model, optimizer=optimizer, epoch=epoch + 1)
        print("Epoch [{}/{}], Average loss : {:.4f}".format(epoch + 1, config["epoch"], np.mean(total_loss)))
        print("모델 저장 완료, 평가")
        evaluate(config, model, bert_tokenizer, epoch)
def save_checkpoint(save_dir, model, optimizer, epoch):
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epoch': epoch}
    filepath = os.path.join(save_dir, 'epoch-checkpoint-{}'.format(epoch))
    torch.save(state_dict, filepath)

def evaluate(config, model, bert_tokenizer, epoch):
    # BERT config 객체 생성
    bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
                                             cache_dir=config["cache_dir_path"])

    # 라벨 딕셔너리 생성
    label2idx, idx2label = read_vocab_data(vocab_data_path=config["label_vocab_data_path"])

    # 평가 데이터 읽기
    test_datas, personal = read_data(file_path=config["test_data_path"], bert_tokenizer=bert_tokenizer)

    # 입력 데이터 전처리
    test_input_ids_features, test_label_id_features = convert_data2feature(datas=test_datas,
                                                                           max_length=config["max_length"],
                                                                           tokenizer=bert_tokenizer,
                                                                           label2idx=label2idx, personal=personal)

    # 평가 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    test_dataset = TensorDataset(test_input_ids_features, test_label_id_features)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"],
                                 sampler=SequentialSampler(test_dataset))

    # 학습한 모델 파일로부터 가중치 불러옴
    model.eval()
    score = 0
    all = 0
    y_true = []
    y_pred = []

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

            input_sequence = []
            until_sep = []
            for i in input_tokens:
                if i == bert_tokenizer.sep_token:
                    input_sequence.append(bert_tokenizer.convert_tokens_to_string(until_sep))
                    until_sep = []
                elif i != '[CLS]':
                    until_sep.append(i)

            # input_sequence = bert_tokenizer.convert_tokens_to_string(
            #     input_tokens[i:input_tokens.index(bert_tokenizer.sep_token)])
            predict = idx2label[hypothesis[index]]
            correct = idx2label[label_id[index]]
            all = all + 1
            if (predict == correct):
                score = score + 1
            # else:
            #     print("입력 : {}".format(input_sequence))
            #     print("출력 : {}, 정답 : {}\n".format(predict, correct))
            y_pred.append(predict)
            y_true.append(correct)

    print(score / all)
    print(score)
    print(all)
    print(classification_report(y_true, y_pred))
    file_path = os.path.join(config["output_dir_path"], "evaluate.txt")
    with open(file_path, "a", encoding="UTF-8-sig") as f:
        f.write("\n\n\n----------------------------\n\n\n")
        f.write("epoch-{}".format(epoch))
        f.write(y)

def test(config):
    # BERT config 객체 생성
    bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
                                             cache_dir=config["cache_dir_path"])
    setattr(bert_config, "num_labels", config["num_labels"])
    # BERT tokenizer 객체 생성
    bert_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
        cache_dir=config["cache_dir_path"])
    setattr(bert_config, "num_labels", config["num_labels"])
    # 라벨 딕셔너리 생성
    label2idx, idx2label = read_vocab_data(vocab_data_path=config["label_vocab_data_path"])

    # 평가 데이터 읽기
    test_datas, personal = read_data(file_path=config["test_data_path"], bert_tokenizer=bert_tokenizer)

    # 입력 데이터 전처리
    test_input_ids_features, test_label_id_features = convert_data2feature(datas=test_datas,
                                                                           max_length=config["max_length"],
                                                                           tokenizer=bert_tokenizer,
                                                                           label2idx=label2idx, personal=personal)

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
    y_true = []
    y_pred = []

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

            input_sequence = []
            until_sep = []
            for i in input_tokens:
                if i == bert_tokenizer.sep_token:
                    input_sequence.append(bert_tokenizer.convert_tokens_to_string(until_sep))
                    until_sep = []
                elif i != '[CLS]':
                    until_sep.append(i)

            # input_sequence = bert_tokenizer.convert_tokens_to_string(
            #     input_tokens[i:input_tokens.index(bert_tokenizer.sep_token)])
            predict = idx2label[hypothesis[index]]
            correct = idx2label[label_id[index]]
            all = all + 1
            if (predict == correct):
                score = score + 1
            # else:
            #     print("입력 : {}".format(input_sequence))
            #     print("출력 : {}, 정답 : {}\n".format(predict, correct))
            y_pred.append(predict)
            y_true.append(correct)

    print(score / all)
    print(score)
    print(all)
    print(classification_report(y_true, y_pred))


if (__name__ == "__main__"):
    output_dir = os.path.join("output_4")
    cache_dir = os.path.join("cache")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    config = {"mode": "train",
              "train_data_path": os.path.join("combined_data_train_10000.json"),
              "test_data_path": os.path.join("combined_data_test_100.json"),
              "output_dir_path": output_dir,
              "cache_dir_path": cache_dir,
              "pretrained_model_name_or_path": "./model/bert-base/",
              "label_vocab_data_path": os.path.join("label_vocab.txt"),
              "num_labels": 9,
              "max_length": 512,
              "epoch": 5,
              "batch_size": 64,
              "checkpoint_path":None
              }

    if (config["mode"] == "train"):
        train(config)
    else:
        test(config)
