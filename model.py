
import torch
import logging

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from anikattu.utilz import LongVar, Var, init_hidden

class Net(nn.Module):
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

        if self.config.HPCONFIG.ACTIVATION == 'softmax':
            attended_question_dist = F.softmax(attended_question_dist, dim=0)
        elif self.config.HPCONFIG.ACTIVATION == 'sigmoid':
            attended_question_dist = F.sigmoid(attended_question_dist)
            
        attended_question_repr = self.__( (attended_question_dist.expand_as(question) * question).sum(dim=0), 'attended_question_repr')
        attended_question_repr = self.__( attended_question_repr.unsqueeze(dim=1), 'attended_question_repr' )

        story_ = self.__( story.transpose(0, 1).transpose(1, 2), 'story_')

        attended_story_dist = self.__( torch.bmm(attended_question_repr, story_), 'attended_story_dist')
        attended_story_dist = self.__( attended_story_dist.transpose(1, 2).transpose(0, 1), 'attended_story_dist')
        
        if self.config.HPCONFIG.ACTIVATION == 'softmax':
            attended_story_dist = F.softmax(attended_story_dist, dim=0)
        elif self.config.HPCONFIG.ACTIVATION == 'sigmoid':
            attended_story_dist = F.sigmoid(attended_story_dist)

        attended_story_repr = (attended_story_dist.expand_as(story) * story).sum(dim=0)
        
        attended_repr = F.tanh(attended_story_repr)
        return (self.__( F.log_softmax(self.answer(attended_repr), dim=-1), 'return val'),
                attended_story_dist.transpose(0,1),
                attended_question_dist.transpose(0,1))
