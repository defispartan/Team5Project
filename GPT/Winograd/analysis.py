import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

#from datasets import _rocstories
from datasets import load_winograd_xml

def rocstories(data_dir, pred_path, log_path):
    preds = pd.read_csv(pred_path, delimiter='\t')['prediction'].values.tolist()

    #_, _, _, labels = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))
   # _, _, _, labels =load_winograd(os.path.join(data_dir, '/mnt/home/panisush/Workspace/842_proj/benchmark/Winograd/test1/test1.c.txt'))
    _, _, _, labels =load_winograd_xml(os.path.join(data_dir, '/mnt/home/panisush/Workspace/842_proj/benchmark/Winograd/test2/WSCollection.xml'))



   #teX1, teX2, teX3, _ =load_winograd_xml(os.path.join(data_dir, '/mnt/home/panisush/Workspace/842_proj/benchmark/Winograd/test2/WSCollection.xml'))


    test_accuracy = accuracy_score(labels, preds)*100.
    logs = [json.loads(line) for line in open(log_path)][1:]
    best_validation_index = np.argmax([log['va_acc'] for log in logs])
    valid_accuracy = logs[best_validation_index]['va_acc']
    print('ROCStories Valid Accuracy: %.2f'%(valid_accuracy))
    print('ROCStories Test Accuracy:  %.2f'%(test_accuracy))
