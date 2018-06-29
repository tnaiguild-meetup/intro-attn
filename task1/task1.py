import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')

import logging
from pprint import pprint, pformat

logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable


from anikattu.utilz import initialize_task


import config

from utilz import Sample, load_task_data
from utilz import PAD,  word_tokenize
from utilz import VOCAB
from utilz import train, predict, plot_attn
from utilz import load_task1_data

from model import Net

SELF_NAME = os.path.basename(__file__).replace('.py', '')
    
import sys
import pickle
if __name__ == '__main__':

    ROOT_DIR = initialize_task(SELF_NAME)

    print('====================================')
    print(ROOT_DIR)
    print('====================================')
        
    if config.CONFIG.flush:
        log.info('flushing...')
        dataset = load_task1_data()
        pickle.dump(dataset, open('{}__cache.pkl'.format(SELF_NAME), 'wb'))
    else:
        dataset = pickle.load(open('{}__cache.pkl'.format(SELF_NAME), 'rb'))
        
    log.info('dataset size: {}'.format(len(dataset.trainset)))
    log.info('dataset[:10]: {}'.format(pformat(dataset.trainset[0])))

    log.info('vocab: {}'.format(pformat(dataset.output_vocab.freq_dict)))
    
    try:
        model =  Net(config, 'Net', len(dataset.input_vocab),  len(dataset.output_vocab))
        model_snapshot = '{}/weights/{}.{}'.format(ROOT_DIR, SELF_NAME, 'pth')
        model.load_state_dict(torch.load(model_snapshot))
        log.info('loaded the old image for the model from :{}'.format(model_snapshot))
    except:
        log.exception('failed to load the model  from :{}'.format(model_snapshot))
        
    if config.CONFIG.cuda:  model = model.cuda()        
    print('**** the model', model)
    
    if 'train' in sys.argv:
        train(config, model, dataset, SELF_NAME, ROOT_DIR)

    if 'predict' in sys.argv:
        print('=========== PREDICTION ==============')
        model.eval()
        count = 0
        while True:
            count += 1
            input_string = input('?')
            if not input_string:
                continue
            
            predict(sys.argv, model, input_string, dataset)
            
            print('Done')
                
