# Copyright (c) Microsoft. All rights reserved.
import json
import pandas as pd
import numpy as np
from random import shuffle
from .label_map import METRIC_FUNC, METRIC_META, METRIC_NAME
from xml.etree import ElementTree
import ast

    

def load_winograd(file, split=0.0):
    rows = []
    cnt = 0
    i = 0
    with open(file, encoding="utf8") as f:
        for line in f.readlines():
            if line.strip() == '': # Skip empty lines
                continue

            if i == 0:
                premise = line.strip()
            elif i == 1:
                special_word = line.strip()
            elif i == 2:
                choices = [c.strip() for c in line.split(',')]
            elif i == 3:
                correct_choice = line.strip()
                if choices[0] == correct_choice:
                    correct_idx = 0
                    olabels = [1, 0]
                elif choices[1] == correct_choice:
                    correct_idx = 1
                    olabels = [0, 1]

                sample = {'uid': cnt, 'premise': premise, 'hypothesis': [premise.replace(special_word, c) for c in choices], 'label': correct_idx, 'ruid': [2*cnt, 2*cnt+1], 'olabel': olabels}
                rows.append(sample)

                cnt += 1
            i += 1
            i %= 4

    if split > 0:
        training_size = int(split*float(len(rows)))
        if training_size % 2 == 1: # Ensure all pairs stay together
            training_size += 1
        return rows[0:training_size], rows[training_size:]
    else:
        return rows

def load_winograd_xml(file, start_id):
    root = ElementTree.parse(file).getroot()

    rows = []
    cnt = start_id # Start with passed in start ID to differentiate the XML Winograd test set which we later concatenate with the Rahman and Ng test set to fit with the training/testing process of BigBird
    for schema in root.findall('schema'):
        text = schema.find('text')
        premise1 = text.find('txt1').text.strip()
        special_word = text.find('pron').text.strip()
        premise2 = text.find('txt2').text.strip()

        answers = schema.find('answers')
        choices = [c.text.strip() for c in answers.findall('answer')]
        i = 0
        for c in choices:
            choices[i] = premise1 + ' ' + c + ' ' + premise2
            i += 1

        correct_idx = schema.find('correctAnswer').text.replace('.', '').strip()
        olabels = []
        if correct_idx == 'A':
            correct_idx = 0
            olabels = [1,0]
        elif correct_idx == 'B':
            correct_idx = 1
            olabels = [0,1]

        sample = {'uid': cnt, 'premise': premise1 + ' ' + special_word + ' ' + premise2, 'hypothesis': choices, 'label': correct_idx, 'ruid': [2 * cnt, 2 * cnt + 1], 'olabel': olabels}

        rows.append(sample)
        cnt += 1

    return rows

def load_copa(file, split=0.0):
    root = ElementTree.parse(file).getroot()

    rows = []
    cnt = 0
    for example in root.findall('item'):
        context = example.find('p').text.strip()
        choices = [example.find('a1').text.strip(), example.find('a2').text.strip()]
        correct_idx = int(example.get('most-plausible-alternative')) - 1
        olabels = [1-correct_idx, correct_idx]

        if example.get('asks-for') == 'cause':
            sample = {'uid': cnt, 'premise': choices, 'hypothesis': context, 'label': correct_idx, 'ruid': [2 * cnt, 2 * cnt + 1], 'olabel': olabels}
        elif example.get('asks-for') == 'effect':
            sample = {'uid': cnt, 'premise': context, 'hypothesis': choices, 'label': correct_idx, 'ruid': [2 * cnt, 2 * cnt + 1], 'olabel': olabels}

        rows.append(sample)
        cnt += 1

    if split > 0:
        training_size = int(split*float(len(rows)))
        if training_size % 2 == 1: # Ensure all pairs stay together
            training_size += 1
        return rows[0:training_size], rows[training_size:]
    else:
        return rows

def load_scitail(file, label_dict):
    """Loading data of scitail
    """
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            blocks = line.strip().split('\t')
            assert len(blocks) > 2
            if blocks[0] == '-': continue
            sample = {'uid': str(cnt), 'premise': blocks[0], 'hypothesis': blocks[1], 'label': label_dict[blocks[2]]}
            rows.append(sample)
            cnt += 1
    return rows

def load_snli(file, label_dict, header=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 10
            if blocks[-1] == '-': continue
            lab = label_dict[blocks[-1]]
            if lab is None:
                import pdb; pdb.set_trace()
            lab = 0 if lab is None else lab
            sample = {'uid': blocks[0], 'premise': blocks[7], 'hypothesis': blocks[8], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_mnli(file, label_dict, header=True, multi_snli=False, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 9
            if blocks[-1] == '-': continue
            lab = 0
            if is_train:
                lab = label_dict[blocks[-1]]
            if lab is None:
                import pdb; pdb.set_trace()
            lab = 0 if lab is None else lab
            sample = {'uid': blocks[0], 'premise': blocks[8], 'hypothesis': blocks[9], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_mrpc(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 4
            lab = 0
            if is_train:
                lab = int(blocks[0])
            sample = {'uid': cnt, 'premise': blocks[-2], 'hypothesis': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_qnli(file, label_dict, header=True, is_train=True):
    """QNLI for classification"""
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 2
            lab = 0
            if is_train:
                lab = label_dict[blocks[-1]]
            if lab is None:
                import pdb; pdb.set_trace()
            lab = 0 if lab is None else lab
            sample = {'uid': blocks[0], 'premise': blocks[1], 'hypothesis': blocks[2], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_qnnli(file, label_dict, header=True, is_train=True):
    """QNLI for ranking"""
    rows = []
    mis_matched_cnt = 0
    cnt = 0
    with open(file, encoding="utf8") as f:
        lines = f.readlines()
        if header: lines = lines[1:]

        assert len(lines) % 2 == 0
        for idx in range(0, len(lines), 2):
            block1 = lines[idx].strip().split('\t')
            block2 = lines[idx + 1].strip().split('\t')
            # train shuffle
            assert len(block1) > 2 and len(block2) > 2
            if is_train and block1[1] != block2[1]:
                mis_matched_cnt += 1
                continue
            assert block1[1] == block2[1]
            lab1, lab2 = 0, 0
            if is_train:
                blocks = [block1, block2]
                shuffle(blocks)
                block1 = blocks[0]
                block2 = blocks[1]
                lab1 = label_dict[block1[-1]]
                lab2 = label_dict[block2[-1]]
                if lab1 == lab2:
                    mis_matched_cnt += 1
                    continue
            lab = int(np.argmax([lab1, lab2]))
            sample = {'uid': cnt, 'premise': block1[1], 'hypothesis': [block1[2], block2[2]], 'label': lab, 'ruid':[block1[0], block2[0]], 'olabel':[lab1, lab2]}
            cnt += 1
            rows.append(sample)
    return rows

def load_qqp(file, header=True, is_train=True):
    rows = []
    cnt = 0
    skipped = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            if is_train and len(blocks) < 6:
                skipped += 1
                continue
            if not is_train: assert len(blocks) == 3
            lab = 0
            if is_train:
                lab = int(blocks[-1])
                sample = {'uid': cnt, 'premise': blocks[-3], 'hypothesis': blocks[-2], 'label': lab}
            else:
                sample = {'uid': int(blocks[0]), 'premise': blocks[-2], 'hypothesis': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_rte(file, label_dict, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header =False
                continue
            blocks = line.strip().split('\t')
            if is_train and len(blocks) < 4: continue
            if not is_train: assert len(blocks) == 3
            lab = 0
            if is_train:
                lab = label_dict[blocks[-1]]
                sample = {'uid': int(blocks[0]), 'premise': blocks[-3], 'hypothesis': blocks[-2], 'label': lab}
            else:
                sample = {'uid': int(blocks[0]), 'premise': blocks[-2], 'hypothesis': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_wnli(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header =False
                continue
            blocks = line.strip().split('\t')
            if is_train and len(blocks) < 4: continue
            if not is_train: assert len(blocks) == 3
            lab = 0
            if is_train:
                lab = int(blocks[-1])
            sample = {'uid': cnt, 'premise': blocks[-2], 'hypothesis': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_diag(file, label_dict, header=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 3
            sample = {'uid': cnt, 'premise': blocks[-3], 'hypothesis': blocks[-2], 'label': label_dict[blocks[-1]]}
            rows.append(sample)
            cnt += 1
    return rows

def load_sst(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            if is_train and len(blocks) < 2: continue
            lab = 0
            if is_train:
                lab = int(blocks[-1])
                sample = {'uid': cnt, 'premise': blocks[0], 'label': lab}
            else:
                sample = {'uid': int(blocks[0]), 'premise': blocks[1], 'label': lab}

            cnt += 1
            rows.append(sample)
    return rows

def load_cola(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            if is_train and len(blocks) < 2: continue
            lab = 0
            if is_train:
                lab = int(blocks[1])
                sample = {'uid': cnt, 'premise': blocks[-1], 'label': lab}
            else:
                sample = {'uid': cnt, 'premise': blocks[-1], 'label': lab}
            rows.append(sample)
            cnt += 1
    return rows

def load_sts(file, header=True, is_train=True):
    rows = []
    cnt = 0
    with open(file, encoding="utf8") as f:
        for line in f:
            if header:
                header = False
                continue
            blocks = line.strip().split('\t')
            assert len(blocks) > 8
            score = 0.0
            if is_train:
                score = float(blocks[-1])
                sample = {'uid': cnt, 'premise': blocks[-3],'hypothesis': blocks[-2], 'label': score}
            else:
                sample = {'uid': cnt, 'premise': blocks[-2],'hypothesis': blocks[-1], 'label': score}
            rows.append(sample)
            cnt += 1
    return rows

def submit(path, data, label_dict=None):
    header = 'index\tprediction'
    with open(path ,'w') as writer:
        predictions, uids = data['predictions'], data['uids']
        writer.write('{}\n'.format(header))
        assert len(predictions) == len(uids)
        # sort label
        paired = [(int(uid), predictions[idx]) for idx, uid in enumerate(uids)]
        paired = sorted(paired, key=lambda item: item[0])
        for uid, pred in paired:
            if label_dict is None:
                writer.write('{}\t{}\n'.format(uid, pred))
            else:
                assert type(pred) is int
                writer.write('{}\t{}\n'.format(uid, label_dict[pred]))

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def eval_model(model, data, dataset, use_cuda=True, with_label=True):
    data.reset()
    if use_cuda:
        model.cuda()
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for batch_meta, batch_data in data:
        score, pred, gold = model.predict(batch_meta, batch_data)
        predictions.extend(pred)
        golds.extend(gold)
        scores.extend(score)
        ids.extend(batch_meta['uids'])
    mmeta = METRIC_META[dataset]
    if with_label:
        for mm in mmeta:
            metric_name = METRIC_NAME[mm]
            metric_func = METRIC_FUNC[mm]
            if mm < 3:
                metric = metric_func(predictions, golds)
            else:
                metric = metric_func(scores, golds)
            metrics[metric_name] = metric
    return metrics, predictions, scores, golds, ids
