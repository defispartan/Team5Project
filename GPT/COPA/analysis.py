import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

#from datasets import _rocstories
from datasets import load_copa

def rocstories(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()

    #_, _, _, labels = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
    _, _, _, labels =load_copa(os.path.join(data_dir, '/mnt/home/panisush/Workspace/842_proj/benchmark/COPA/test/copa-test.xml'))

    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ROCStories Valid Accuracy: %.2f'%(valid_accuracy))
    print('ROCStories Test Accuracy:  %.2f'%(test_accuracy))

data_dir='/mnt/home/panisush/Workspace/842_proj/finetune-transformer-lm-master/'
log_dir='/mnt/home/panisush/Workspace/842_proj/finetune-transformer-lm-master/log/'
submission_dir='/mnt/home/panisush/Workspace/842_proj/finetune-transformer-lm-master/submission/'



def predict():
    filename = filenames[dataset]
    pred_fn = pred_fns[dataset]
    label_decoder = label_decoders[dataset]
    predictions = pred_fn(iter_predict(teX, teM))
    if label_decoder is not None:
        predictions = [label_decoder[prediction] for prediction in predictions]
    path = os.path.join(submission_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('{}\t{}\n'.format('index', 'prediction'))
        for i, prediction in enumerate(predictions):
            f.write('{}\t{}\n'.format(i, prediction))

sess.run([p.assign(ip) for p, ip in zip(params, joblib.load(os.path.join(save_dir, desc, 'best_params.jl')))])
predict()
rocstories(data_dir, os.path.join(submission_dir, 'ROCStories.tsv'), os.path.join(log_dir, 'rocstories.jsonl'))
