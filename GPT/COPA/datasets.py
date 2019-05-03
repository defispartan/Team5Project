import os
import csv
import numpy as np
from xml.etree import ElementTree


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445




def rocstories(data_dir, n_train=1497, n_valid=374):
    print("Sushant *************************************************************",data_dir)

    #storys, comps1, comps2, ys = _rocstories(os.path.join(data_dir, '842_proj/benchmark/copa/train'))
    # teX1, teX2, teX3, _ = _rocstories(os.path.join(data_dir, 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'))

    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = load_copa(os.path.join(data_dir, '/mnt/home/panisush/Workspace/842_proj/benchmark/COPA/train/copa-train.xml'), split=0.90)

    teX1, teX2, teX3, _ =load_copa(os.path.join(data_dir, '/mnt/home/panisush/Workspace/842_proj/benchmark/COPA/test/copa-test.xml'))
   
   # tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)



def load_copa(file, split=0.0):
    root = ElementTree.parse(file).getroot()

    #rows = []

    choice1=[]
    choice2=[]
    label=[]
    context=[]


    #cnt = 0

    for example in root.findall('item'):
        c=example.find('p').text.strip()
        #context.append(example.find('p').text.strip())
        choice1.append(example.find('a1').text.strip())
        choice2.append(example.find('a2').text.strip())  

        #correct_idx = int(example.get('most-plausible-alternative')) - 1
        label.append(int(example.get('most-plausible-alternative')) - 1)

        #olabels = [1-correct_idx, correct_idx]

        if example.get('asks-for') == 'cause':
            c=c+' what was the cause of this?'
            #sample = {'uid': cnt, 'premise': choices, 'hypothesis': context, 'label': correct_idx, 'ruid': [2 * cnt, 2 * cnt + 1], 'olabel': olabels}
        elif example.get('asks-for') == 'effect':
            c=c+' what happened as a result?'
            #sample = {'uid': cnt, 'premise': context, 'hypothesis': choices, 'label': correct_idx, 'ruid': [2 * cnt, 2 * cnt + 1], 'olabel': olabels}

        context.append(c)
        #cnt += 1


	#return context, choice1, choice2

    if split > 0:
        training_size = int(split*float(len(context)))
       # if training_size % 2 == 1: # Ensure all pairs stay together
        #    training_size += 1
        return context[0:training_size], context[training_size:], choice1[0:training_size], choice1[training_size:], choice2[0:training_size], choice2[training_size:], label[0:training_size], label[training_size:]
    else:
        return context, choice1, choice2, label

