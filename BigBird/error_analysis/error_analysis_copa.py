import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
import json

def copa_label_to_int(l):
    if l == 'implausible':
        return 0
    else:
        return 1

copa_pred = pd.read_csv('copa_pred.tsv', delim_whitespace=True, lineterminator='\n', header='infer')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

incorrect_samples = []
correct_samples = []
with open('copa_test.json', 'r', encoding='utf-8') as reader:
    data = []
    cnt = 0
    incorrect_cnt = 0
    correct_cnt = 0
    for line in reader:
        sample = json.loads(line)

        p1 = copa_label_to_int(copa_pred.iloc[2*cnt]['prediction'])
        p2 = copa_label_to_int(copa_pred.iloc[2*cnt+1]['prediction'])

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

print(str(incorrect_cnt) + ' incorrect answers found')
print(str(correct_cnt) + ' correct answers found')
print(str(correct_cnt/cnt) + ' accuracy')

with open('copa_incorrect.json', 'w') as fout:
    for d in incorrect_samples:
        json.dump(d, fout)
        fout.write('\n')

with open('copa_correct.json', 'w') as fout:
    for d in correct_samples:
        json.dump(d, fout)
        fout.write('\n')