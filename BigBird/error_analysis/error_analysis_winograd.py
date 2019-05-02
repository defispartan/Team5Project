# Some data indexing issues happened in preprocessing for Winograd, so there are some hacky things happening here to sort that out...

import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
import json

with open('winograd_pred.json', 'r', encoding='utf-8') as reader:
    data = json.loads(reader.readline())
    predictions = data['predictions']
    ids = data['uids']
winograd_pred_1 = []
winograd_pred_2 = []
last_id = 0
first_dataset = True
i = 0
for idd in ids:
    if idd < last_id: 
        first_dataset = False
    
    if first_dataset:
        winograd_pred_1.append(predictions[i])
    else:
        winograd_pred_2.append(predictions[i])

    last_id = idd
    i += 1

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

incorrect_samples = []
correct_samples = []
with open('winograd_test1.json', 'r', encoding='utf-8') as reader:
    data = []
    cnt = 0
    incorrect_cnt = 0
    correct_cnt = 0
    for line in reader:
        sample = json.loads(line)

        p1 = winograd_pred_1[2*cnt]
        p2 = winograd_pred_1[2*cnt+1]

        t1 = sample['olabel'][0]
        t2 = sample['olabel'][1]

        new_sample = {}
        new_sample['gt'] = sample['olabel']
        new_sample['pred'] = [p1,p2]
        new_sample['tokens1'] = bert_tokenizer.convert_ids_to_tokens(sample['token_id'][0])
        new_sample['tokens2'] = bert_tokenizer.convert_ids_to_tokens(sample['token_id'][1])

        cnt += 1
        if p1==t1 and p2==t2:
            correct_samples.append(new_sample)
            correct_cnt += 1
        else:
            incorrect_samples.append(new_sample)
            incorrect_cnt += 1

print('TEST SET 1')
print(str(incorrect_cnt) + ' incorrect answers found')
print(str(correct_cnt) + ' correct answers found')
print(str(correct_cnt/cnt) + ' accuracy')

with open('winograd_incorrect_1.json', 'w') as fout:
    for d in incorrect_samples:
        json.dump(d, fout)
        fout.write('\n')

with open('winograd_correct_1.json', 'w') as fout:
    for d in correct_samples:
        json.dump(d, fout)
        fout.write('\n')

incorrect_samples = []
correct_samples = []
with open('winograd_test2.json', 'r', encoding='utf-8') as reader:
    data = []
    cnt_2 = 0
    incorrect_cnt = 0
    correct_cnt = 0
    for line in reader:
        sample = json.loads(line)

        p1 = winograd_pred_2[2*cnt_2]
        p2 = winograd_pred_2[2*cnt_2+1]

        t1 = sample['olabel'][0]
        t2 = sample['olabel'][1]

        new_sample = {}
        new_sample['gt'] = sample['olabel']
        new_sample['pred'] = [p1,p2]
        new_sample['tokens1'] = bert_tokenizer.convert_ids_to_tokens(sample['token_id'][0])
        new_sample['tokens2'] = bert_tokenizer.convert_ids_to_tokens(sample['token_id'][1])

        cnt += 1
        cnt_2 += 1
        if p1==t1 and p2==t2:
            correct_samples.append(new_sample)
            correct_cnt += 1
        else:
            incorrect_samples.append(new_sample)
            incorrect_cnt += 1

print('TEST SET 2')
print(str(incorrect_cnt) + ' incorrect answers found')
print(str(correct_cnt) + ' correct answers found')
print(str(correct_cnt/cnt_2) + ' accuracy')

with open('winograd_incorrect_2.json', 'w') as fout:
    for d in incorrect_samples:
        json.dump(d, fout)
        fout.write('\n')

with open('winograd_correct_2.json', 'w') as fout:
    for d in correct_samples:
        json.dump(d, fout)
        fout.write('\n')