import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')

from anikattu.logger import CMDFilter
import logging
from pprint import pprint, pformat

logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import config

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.trainer import Trainer, Tester, Predictor
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.utilz import initialize_task
from functools import partial

from collections import namedtuple, defaultdict, Counter
import itertools

from utilz import MacNetSample as Sample
from utilz import PAD,  word_tokenize
from utilz import VOCAB
from anikattu.utilz import pad_seq

from anikattu.utilz import logger
from anikattu.vocab import Vocab
from anikattu.tokenstring import TokenString
from anikattu.utilz import LongVar, Var, init_hidden
import numpy as np

import glob

SELF_NAME = os.path.basename(__file__).replace('.py', '')

def load_task_data(task=1, type_='train', max_sample_size=None):
    samples = []
    qn, an = 0, 0
    skipped = 0

    input_vocabulary = Counter()
    output_vocabulary = Counter()
    
    try:
        filename = glob.glob('../dataset/en-10k/qa{}_*_{}.txt'.format(task, type_))[0]
              
        task_name = re.search(r'qa\d+_(.*)_.*.txt', filename)
        if task_name:
            task_name = task_name.group(1)
            
        log.info('processing file: {}'.format(filename))
        dataset = open(filename).readlines()
        prev_linenum = 1000000
        for line in tqdm(dataset):
            questions, answers = [], []
            linenum, line = line.split(' ', 1)

            linenum = int(linenum)
            if prev_linenum > linenum:
                story = ''

            if '?' in line:
                q, a, _ = line.split('\t')

                samples.append(
                    Sample('{}.{}'.format(task, linenum),
                           task, linenum,
                           task_name,
                           TokenString(story.lower(), word_tokenize),
                           TokenString(q.lower(),     word_tokenize),
                           a.lower())
                    )

            else:
                story += ' ' + line

            prev_linenum = linenum

    except:
        skipped += 1
        log.exception('{}'.format(task, linenum))
        
    print('skipped {} samples'.format(skipped))
    
    samples = sorted(samples, key=lambda x: len(x.story), reverse=True)
    if max_sample_size:
        samples = samples[:max_sample_size]

    log.info('building input_vocabulary...')
    for sample in samples:
        input_vocabulary.update(sample.story + sample.q)            
        output_vocabulary.update([sample.a])

    return task_name, samples, input_vocabulary, output_vocabulary


def load_data(max_sample_size=None):
    dataset = {}
    task_name, train_samples, train_input_vocab, train_output_vocab = load_task_data(task=1, type_='train')
    task_name, test_samples, test_input_vocab, test_output_vocab = load_task_data(task=1, type_='test')

    input_vocab = train_input_vocab + test_input_vocab
    output_vocab = train_output_vocab + test_output_vocab

    return Dataset(task_name, (train_samples, test_samples), Vocab(input_vocab, special_tokens=VOCAB), Vocab(output_vocab))        

# ## Loss and accuracy function
def loss(output, batch, loss_function, *args, **kwargs):
    indices, (story, question), (answer) = batch
    output, attn = output
    return loss_function(output, answer)

def accuracy(output, batch, *args, **kwargs):
    indices, (story, question), (answer) = batch
    output, attn = output
    return (output.max(dim=1)[1] == answer).sum().float()/float(answer.size(0))

def repr_function(output, batch, VOCAB, LABELS, dataset):
    indices, (story, question), (answer) = batch
    
    results = []
    output, attn = output
    output = output.max(1)[1]
    output = output.cpu().numpy()
    for idx, c, q, a, o in zip(indices, story, question, answer, output):

        c = ' '.join([VOCAB[i] for i in c]).replace('\n', ' ')
        q = ' '.join([VOCAB[i] for i in q])
        a = ' '.join([LABELS[a]])
        o = ' '.join([LABELS[o]])
        
        results.append([idx, dataset[idx].task_name, c, q, a, o, str(a == o) ])
        
    return results


def batchop(datapoints, VOCAB, LABELS, *args, **kwargs):
    indices = [d.id for d in datapoints]
    story = []
    question = []
    answer = []

    for d in datapoints:
        story.append([VOCAB[w] for w in d.story])
        question.append([VOCAB[w] for w in d.q])
        answer.append(LABELS[d.a])

    story    = LongVar(pad_seq(story))
    question = LongVar(pad_seq(question))
    answer   = LongVar(answer)

    batch = indices, (story, question), (answer)
    return batch

def predict_batchop(datapoints, VOCAB, LABELS, *args, **kwargs):
    indices = [d.id for d in datapoints]
    story = []
    question = []

    for d in datapoints:
        story.append([VOCAB[w] for w in d.story])
        question.append([VOCAB[w] for w in d.q])

    story    = LongVar(pad_seq(story))
    question = LongVar(pad_seq(question))

    batch = indices, (story, question), ()
    return batch
                          
class MacNet(nn.Module):
    def __init__(self, config, name, input_vocab_size, output_vocab_size):
        
        super().__init__()
        self._name = name

        size_log_name = '{}.{}'.format(self._name, 'size')
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.setLevel(logging.INFO)
        self.print_instance = 0
        
        self.config = config

        self.embed_size = config.HPCONFIG.embed_size
        self.hidden_size = config.HPCONFIG.hidden_size
        
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        
        self.embed = nn.Embedding(self.input_vocab_size, self.embed_size)
        
        self.encode_story    = nn.GRU(self.embed_size, self.hidden_size, bidirectional=True)
        self.encode_question = nn.GRU(self.embed_size, self.hidden_size, bidirectional=True)

        self.attend_story    = nn.Linear(2 * self.hidden_size, 1)
        self.attend_question = nn.Linear(2 * self.hidden_size, 1)

        self.answer = nn.Linear(2 * self.hidden_size, self.output_vocab_size)

        if config.CONFIG.cuda:
             self.cuda()
             
        
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
            if self.print_instance or print_instance:
                self.size_log.debug(tensor)

            
        return tensor

    def name(self, n):
        return '{}.{}'.format(self._name, n)

    def forward(self, input_):
        idxs, inputs, targets = input_
        story, question = inputs
        story = self.__( story, 'story')
        question = self.__(question, 'question')

        batch_size, story_size  = story.size()
        batch_size, question_size = question.size()
        
        story  = self.__( self.embed(story),  'story_emb')
        question = self.__( self.embed(question), 'question_emb')

        story  = story.transpose(1,0)
        story, _  = self.__(  self.encode_story(
            story,
            init_hidden(batch_size, self.encode_story)), 'C'
        )
        
        question  = question.transpose(1,0)
        question, _ = self.__(  self.encode_question(
            question,
            init_hidden(batch_size, self.encode_question)), 'Q'
        )

        story = F.tanh(story)
        question = F.tanh(question)

        attended_question_dist = self.__( self.attend_question(question.view(-1, 2 * self.hidden_size)), 'attended_question_dist')
        attended_question_dist = self.__( attended_question_dist.view(question_size, batch_size, 1), 'attended_question_dist')
        attended_question_repr = self.__( (attended_question_dist.expand_as(question) * question).sum(dim=0), 'attended_question_repr')
        attended_question_repr = self.__( attended_question_repr.unsqueeze(dim=1), 'attended_question_repr' )

        story_ = self.__( story.transpose(0, 1).transpose(1, 2), 'story_')

        attended_story_dist = self.__( torch.bmm(attended_question_repr, story_), 'attended_story_dist')
        attended_story_dist = self.__( attended_story_dist.transpose(1, 2).transpose(0, 1), 'attended_story_dist')
        attended_story_dist = F.sigmoid(attended_story_dist)
        attended_story_repr = (attended_story_dist.expand_as(story) * story).sum(dim=0)
        
        attended_repr = F.tanh(attended_story_repr)
        return self.__( F.log_softmax(self.answer(attended_repr), dim=-1), 'return val'), attended_story_dist.transpose(0,1)
            
    
import sys
import pickle
if __name__ == '__main__':

    if sys.argv[1]:
        log.addFilter(CMDFilter(sys.argv[1]))

    ROOT_DIR = initialize_task(SELF_NAME)

    print('====================================')
    print(ROOT_DIR)
    print('====================================')
        
    if config.CONFIG.flush:
        log.info('flushing...')
        dataset = load_data()
        pickle.dump(dataset, open('{}__cache.pkl'.format(SELF_NAME), 'wb'))
    else:
        dataset = pickle.load(open('{}__cache.pkl'.format(SELF_NAME), 'rb'))
        
    log.info('dataset size: {}'.format(len(dataset.trainset)))
    log.info('dataset[:10]: {}'.format(pformat(dataset.trainset[0])))

    log.info('vocab: {}'.format(pformat(dataset.output_vocab.freq_dict)))
    
    try:
        model =  MacNet(config, 'macnet', len(dataset.input_vocab),  len(dataset.output_vocab))
        model_snapshot = '{}/weights/{}.{}'.format(ROOT_DIR, SELF_NAME, 'pth')
        model.load_state_dict(torch.load(model_snapshot))
        log.info('loaded the old image for the model from :{}'.format(model_snapshot))
    except:
        log.exception('failed to load the model  from :{}'.format(model_snapshot))
        
    if config.CONFIG.cuda:  model = model.cuda()        
    print('**** the model', model)
    
    if 'train' in sys.argv:
        _batchop = partial(batchop, VOCAB=dataset.input_vocab, LABELS=dataset.output_vocab)
        train_feed     = DataFeed(SELF_NAME, dataset.trainset, batchop=_batchop, batch_size=config.HPCONFIG.batch_size)
        predictor_feed = DataFeed(SELF_NAME, dataset.testset, batchop=_batchop, batch_size=1)

        predictor = Predictor(SELF_NAME,
                              model=model,
                              directory=ROOT_DIR,
                              feed=predictor_feed,
                              repr_function=partial(repr_function
                                                    , VOCAB=dataset.input_vocab
                                                    , LABELS=dataset.output_vocab
                                                    , dataset=dataset.testset_dict))

        
        loss_ = partial(loss, loss_function=nn.NLLLoss())
        test_feed     = DataFeed(SELF_NAME, dataset.testset, batchop=_batchop, batch_size=config.HPCONFIG.batch_size)

        tester = Tester(name  = SELF_NAME,
                                      config   = config,
                                      model    = model,
                                      directory = ROOT_DIR,
                                      loss_function = loss_,
                                      accuracy_function = accuracy,
                                      feed = test_feed,
                                      predictor=predictor)

        trainer = Trainer(name=SELF_NAME,
                          config = config,
                          model=model,
                          directory=ROOT_DIR,
                          optimizer  = optim.Adam(model.parameters()),
                          loss_function = loss_,
                          checkpoint = config.CONFIG.CHECKPOINT,
                          do_every_checkpoint = tester.do_every_checkpoint,
                          epochs = config.CONFIG.EPOCHS,
                          feed = train_feed,
        )
        

        
        for e in range(config.CONFIG.EONS):
            
            if not trainer.train():
                raise Exception

            dump = open('{}/results/eon_{}.csv'.format(ROOT_DIR, e), 'w')
            log.info('on {}th eon'.format(e))
            results = ListTable()
            for ri in tqdm(range(predictor_feed.num_batch)):
                output, _results = predictor.predict(ri)
                results.extend(_results)
            dump.write(repr(results))
            dump.close()

    if 'predict' in sys.argv:
        print('=========== PREDICTION ==============')
        model.eval()
        count = 0
        while True:
            count += 1
            sentence = []
            input_string = input('?')
            if not input_string:
                continue
            
            story, question = input_string.lower().split('|') 
            story_tokens = word_tokenize(story)
            question_ = word_tokenize(question)

            input_ = predict_batchop(
                datapoints = [Sample('0', 'story 1', 'question 1', 'task 1', story_tokens, question_, '')],
                VOCAB      = dataset.input_vocab,
                LABELS     = dataset.output_vocab
            )
            
            output, attn = model(input_)
            attn = attn.squeeze().data.cpu()
            print('attn', attn.size())
            print('story_tokens', len(story_tokens))

            answer = dataset.output_vocab[output.max(1)[1]]
            print(answer)

            if 'show_plot' in sys.argv or 'save_plot' in sys.argv:

                from matplotlib import pyplot as plt
                plt.figure(figsize=(20,10))

                pprint(list(zip(story_tokens, attn.tolist())))
                nwords = len(story_tokens)
                plt.bar(range(nwords), attn.tolist())
                plt.title('{}\n{}'.format(question, answer))
                plt.xticks(range(nwords), story_tokens, rotation='vertical')

                if 'show_plot' in sys.argv:
                    plt.show()
                if 'save_plot' in sys.argv:
                    plt.savefig('{}.png'.format(count))
                plt.close()

            print('Done')
                
