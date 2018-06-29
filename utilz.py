from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from functools import partial

import re
import glob
from collections import namedtuple, defaultdict, Counter


from anikattu.trainer import Trainer, Tester, Predictor
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.datafeed import DataFeed, MultiplexedDataFeed

from anikattu.tokenizer import word_tokenize
from anikattu.vocab import Vocab

from anikattu.utilz import tqdm, ListTable
from anikattu.utilz import pad_seq

from anikattu.tokenstring import TokenString
from anikattu.utilz import LongVar, Var, init_hidden
import numpy as np

from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize


VOCAB =  ['PAD', 'UNK', 'GO', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""
Sample   =  namedtuple('Sample', ['id', 'aid', 'qid', 'task_name', 'story', 'q', 'a'])


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

def load_task1_data(max_sample_size=None):
    task_name, train_samples, train_input_vocab, train_output_vocab = load_task_data(task=1, type_='train')
    task_name, test_samples, test_input_vocab, test_output_vocab = load_task_data(task=1, type_='test')

    input_vocab = train_input_vocab + test_input_vocab
    output_vocab = train_output_vocab + test_output_vocab

    return Dataset(task_name, (train_samples, test_samples), Vocab(input_vocab, special_tokens=VOCAB), Vocab(output_vocab))        

def load_task6_data(max_sample_size=None):
    task_name, train_samples, train_input_vocab, train_output_vocab = load_task_data(task=6, type_='train')
    task_name, test_samples, test_input_vocab, test_output_vocab = load_task_data(task=6, type_='test')

    input_vocab = train_input_vocab + test_input_vocab
    output_vocab = train_output_vocab + test_output_vocab

    return Dataset(task_name, (train_samples, test_samples), Vocab(input_vocab, special_tokens=VOCAB), Vocab(output_vocab))        

def load_task1_task6_data(max_sample_size=None):
    trainset, testset = [], []
    input_vocab, output_vocab = Counter(), Counter()
    for i in [1, 6]:
        task_name, train_samples, train_input_vocab, train_output_vocab = load_task_data(task=i, type_='train')
        task_name, test_samples, test_input_vocab, test_output_vocab = load_task_data(task=i, type_='test')

        trainset += train_samples
        testset += test_samples
        input_vocab += train_input_vocab + test_input_vocab
        output_vocab += train_output_vocab + test_output_vocab

    return Dataset(task_name, (trainset, testset), Vocab(input_vocab, special_tokens=VOCAB), Vocab(output_vocab))        


# ## Loss and accuracy function
def loss(output, batch, loss_function, *args, **kwargs):
    indices, (story, question), (answer) = batch
    output, sattn, qattn = output
    return loss_function(output, answer)

def accuracy(output, batch, *args, **kwargs):
    indices, (story, question), (answer) = batch
    output, sattn, qattn = output
    return (output.max(dim=1)[1] == answer).sum().float()/float(answer.size(0))

def repr_function(output, batch, VOCAB, LABELS, dataset):
    indices, (story, question), (answer) = batch
    
    results = []
    output, sattn, qattn = output
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

def train(config, model, dataset,  SELF_NAME, ROOT_DIR, batchop=batchop, loss=loss, accuracy=accuracy, repr_function=repr_function):
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
        
def predict(argv, model, input_string, dataset):
                
    story, question = input_string.lower().split('|') 
    story_tokens = word_tokenize(story)
    question_tokens = word_tokenize(question)
    
    input_ = predict_batchop(
        datapoints = [Sample('0', 'story 1', 'question 1', 'task 1', story_tokens, question_tokens, '')],
        VOCAB      = dataset.input_vocab,
        LABELS     = dataset.output_vocab
    )
            
    output = model(input_)
    plot_attn(argv, question_tokens, story_tokens, output, dataset)
    
def plot_attn(argv, question_tokens, story_tokens, output, dataset):
    output, sattn, qattn = output
    sattn = sattn.squeeze().data.cpu()
    print('sattn', sattn.size())
    qattn = qattn.squeeze().data.cpu()
    print('qattn', qattn.size())
    print('story_tokens', len(story_tokens))

    answer = dataset.output_vocab[output.max(1)[1]]
    print(answer)
    if 'show_plot' in argv or 'save_plot' in argv:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(20,10))
        
        nwords = len(question_tokens)
        plt.bar(range(nwords), qattn.tolist())
        plt.title('{}\n{}'.format(' '.join(question_tokens), answer))
        plt.xticks(range(nwords), question_tokens, rotation='vertical')
        
        if 'show_plot' in argv:
            plt.show()
        if 'save_plot' in argv:
            plt.savefig('{}-story.png'.format(count))
        plt.close()
        
        nwords = len(story_tokens)
        plt.bar(range(nwords), sattn.tolist())
        plt.title('{}\n{}'.format(' '.join(question_tokens), answer))
        plt.xticks(range(nwords), story_tokens, rotation='vertical')
        
        if 'show_plot' in argv:
            plt.show()
        if 'save_plot' in argv:
            plt.savefig('{}-story.png'.format(count))
        plt.close()
