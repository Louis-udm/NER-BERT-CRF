# -*- coding: utf-8 -*-

# # # #
# NER_BERT_CRF.py
# @author Zhibin.LU
# @created Fri Feb 15 2019 22:47:19 GMT-0500 (EST)
# @last-modified Sun Mar 31 2019 12:17:08 GMT-0400 (EDT)
# @website: https://louis-udm.github.io
# @description: Bert pytorch pretrainde model with or without CRF for NER
# The NER_BERT_CRF.py include 2 model:
# - model 1:
#   - This is just a pretrained BertForTokenClassification, For a comparision with my BERT-CRF model
# - model 2:
#   - A pretrained BERT with CRF model.
# - data set
#   - [CoNLL-2003](https://github.com/FuYanzhe2/Name-Entity-Recognition/tree/master/BERT-BiLSTM-CRF-NER/NERdata)
# # # #


# %%
import sys
import os
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

from torch.utils.data.distributed import DistributedSampler
from torch.utils import data

from tqdm import tqdm, trange
import collections

from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
import pickle
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer

def set_work_dir(local_path="ner_bert_crf", server_path="ner_bert_crf"):
    if (os.path.exists(os.getenv("HOME")+'/'+local_path)):
        os.chdir(os.getenv("HOME")+'/'+local_path)
    elif (os.path.exists(os.getenv("HOME")+'/'+server_path)):
        os.chdir(os.getenv("HOME")+'/'+server_path)
    else:
        raise Exception('Set work path error!')


def get_data_dir(local_path="ner_bert_crf", server_path="ner_bert_crf"):
    if (os.path.exists(os.getenv("HOME")+'/'+local_path)):
        return os.getenv("HOME")+'/'+local_path
    elif (os.path.exists(os.getenv("HOME")+'/'+server_path)):
        return os.getenv("HOME")+'/'+server_path
    else:
        raise Exception('get data path error!')


print('Python version ', sys.version)
print('PyTorch version ', torch.__version__)

set_work_dir()
print('Current dir:', os.getcwd())

cuda_yes = torch.cuda.is_available()
# cuda_yes = False
print('Cuda is available?', cuda_yes)
device = torch.device("cuda:0" if cuda_yes else "cpu")
print('Device:', device)

data_dir = os.path.join(get_data_dir(), 'NER_data/CoNLL2003/')
# "Whether to run training."
do_train = True
# "Whether to run eval on the dev set."
do_eval = True
# "Whether to run the model in inference mode on the test set."
do_predict = True
# "The vocabulary file that the BERT model was trained on."
max_seq_length = 150 #128
batch_size = 32 #32
# "The initial learning rate for Adam."
learning_rate0 = 5e-5
lr0_crf_fc = 8e-5
weight_decay_finetune = 1e-5 #0.01
weight_decay_crf_fc = 5e-6 #0.005
total_train_epochs = 15
gradient_accumulation_steps = 1
warmup_proportion = 0.1
output_dir = './output/'
bert_model_scale = 'bert-base-cased'
do_lower_case = False
# eval_batch_size = 8
# predict_batch_size = 8
# "Proportion of training to perform linear learning rate warmup for. "
# "E.g., 0.1 = 10% of training."
warmup_proportion = 0.1
# "How often to save the model checkpoint."
# save_checkpoints_steps = 1000
# "How many steps to make in each estimator call."
# iterations_per_loop = 1000
load_checkpoint = True


# %%
'''
Functions and Classes for read and organize data set
'''

class InputExample(object):
    """A single training/test example for NER."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example(a sentence or a pair of sentences).
          words: list of words of sentence
          labels_a/labels_b: (Optional) string. The label seqence of the text_a/text_b. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data.
    result of convert_examples_to_features(InputExample)
    """

    def __init__(self, input_ids, input_mask, segment_ids,  predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """
        Reads a BIO data.
        """
        with open(input_file) as f:
            # out_lines = []
            out_lists = []
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                words = []
                ner_labels = []
                pos_tags = []
                bio_pos_tags = []
                for line in entry.splitlines():
                    pieces = line.strip().split()
                    if len(pieces) < 1:
                        continue
                    word = pieces[0]
                    # if word == "-DOCSTART-" or word == '':
                    #     continue
                    words.append(word)
                    pos_tags.append(pieces[1])
                    bio_pos_tags.append(pieces[2])
                    ner_labels.append(pieces[-1])
                # sentence = ' '.join(words)
                # ner_seq = ' '.join(ner_labels)
                # pos_tag_seq = ' '.join(pos_tags) 
                # bio_pos_tag_seq = ' '.join(bio_pos_tags) 
                # out_lines.append([sentence, pos_tag_seq, bio_pos_tag_seq, ner_seq])
                # out_lines.append([sentence, ner_seq])
                out_lists.append([words,pos_tags,bio_pos_tags,ner_labels])
        return out_lists


class CoNLLDataProcessor(DataProcessor):
    '''
    CoNLL-2003
    '''

    def __init__(self):
        self._label_types = [ 'X', '[CLS]', '[SEP]', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG']
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i,
                           label in enumerate(self._label_types)}

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "valid.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map
    
    def get_start_label_id(self):
        return self._label_map['CLS']

    def get_stop_label_id(self):
        return self._label_map['SEP']

    def _create_examples(self, all_lists):
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            words = one_lists[0]
            labels = one_lists[-1]
            examples.append(InputExample(
                guid=guid, words=words, labels=labels))
        return examples

    def _create_examples2(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text = line[0]
            ner_label = line[-1]
            examples.append(InputExample(
                guid=guid, text_a=text, labels_a=ner_label))
        return examples


def example2feature(example, tokenizer, label_map, max_seq_length):

    add_label = 'X'
    # tokenize_count = []
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['CLS']]
    for i, w in enumerate(example.words):
        # use bertTokenizer to split words
        # 1996-08-22 => 1996 - 08 - 22
        # sheepmeat => sheep ##me ##at
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        # tokenize_count.append(len(sub_words))
        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                # '##xxx' -> 'X' (see bert paper)
                predict_mask.append(0)
                label_ids.append(label_map[add_label])

    # truncate
    if len(tokens) > max_seq_length - 1:
        print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens), max_seq_length))
        tokens = tokens[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
    tokens.append('[SEP]')
    predict_mask.append(0)
    label_ids.append(label_map['SEP'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    feat=InputFeatures(
                # guid=example.guid,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                predict_mask=predict_mask,
                label_ids=label_ids)

    return feat

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples=examples
        self.tokenizer=tokenizer
        self.label_map=label_map
        self.max_seq_length=max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat=example2feature(self.examples[idx], self.tokenizer, self.label_map, max_seq_length)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids

def pad(batch):

    seqlen_list = [len(sample[0]) for sample in batch]
    maxlen = np.array(seqlen_list).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: X for padding
    input_ids_list = torch.LongTensor(f(0, maxlen))
    input_mask_list = torch.LongTensor(f(1, maxlen))
    segment_ids_list = torch.LongTensor(f(2, maxlen))
    predict_mask_list = torch.ByteTensor(f(3, maxlen))
    label_ids_list = torch.LongTensor(f(4, maxlen))

    return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list

def f1_score(y_true, y_pred):
    '''
    0,1,2,3 are [CLS],[SEP],[X],O
    '''
    ignore_id=3
    
    num_proposed = len(y_pred[y_pred>ignore_id])
    num_correct = (np.logical_and(y_true==y_pred, y_true>ignore_id)).sum()
    num_gold = len(y_true[y_true>ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    return precision, recall, f1

#%%
'''
Prepare data set
'''
# random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
if cuda_yes:
    torch.cuda.manual_seed_all(44)

# Load pre-trained model tokenizer (vocabulary)
conllProcessor = CoNLLDataProcessor()
label_list = conllProcessor.get_labels()
label_map = conllProcessor.get_label_map()
train_examples = conllProcessor.get_train_examples(data_dir)
dev_examples = conllProcessor.get_dev_examples(data_dir)
test_examples = conllProcessor.get_test_examples(data_dir)

total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)

print("***** Running training *****")
print("  Num examples = %d"% len(train_examples))
print("  Batch size = %d"% batch_size)
print("  Num steps = %d"% total_train_steps)

tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

train_dataset = NerDataset(train_examples,tokenizer,label_map,max_seq_length)
dev_dataset = NerDataset(dev_examples,tokenizer,label_map,max_seq_length)
test_dataset = NerDataset(test_examples,tokenizer,label_map,max_seq_length)

train_dataloader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=pad)

dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

test_dataloader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)


#%%
'''
#####  Use only BertForTokenClassification  #####
'''
print('*** Use only BertForTokenClassification ***')

if load_checkpoint and os.path.exists(output_dir+'/ner_bert_checkpoint.pt'):
    checkpoint = torch.load(output_dir+'/ner_bert_checkpoint.pt', map_location='cpu')
    start_epoch = checkpoint['epoch']+1
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    model = BertForTokenClassification.from_pretrained(
        bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))
    print('Loaded the pretrain NER_BERT model, epoch:',checkpoint['epoch'],'valid acc:', 
            checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
else:
    start_epoch = 0
    valid_acc_prev = 0
    valid_f1_prev = 0
    model = BertForTokenClassification.from_pretrained(
        bert_model_scale, num_labels=len(label_list))

model.to(device)

# Prepare optimizer
named_params = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate0)

def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    # print("***** Running prediction *****")
    model.eval()
    all_preds = []
    all_labels = []
    total=0
    correct=0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            out_scores = model(input_ids, segment_ids, input_mask)
            # out_scores = out_scores.detach().cpu().numpy()
            _, predicted = torch.max(out_scores, -1)
            valid_predicted = torch.masked_select(predicted, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct/total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend: %.3f minutes for evaluation' \
        % (epoch_th, 100.*test_acc, 100.*precision, 100.*recall, 100.*f1, dataset_name,(end-start)/60.0))
    print('--------------------------------------------------------------')
    return test_acc, f1


#%%
# train procedure using only BertForTokenClassification
train_start = time.time()
global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)
# for epoch in trange(start_epoch, total_train_epochs, desc="Epoch"):
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = learning_rate0 * warmup_linear(global_step_th/total_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
          
        print("Epoch:{}-{}/{}, CrossEntropyLoss: {} ".format(epoch, step, len(train_dataloader), loss.item()))
    
    print('--------------------------------------------------------------')
    print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss, (time.time() - train_start)/60.0))
    valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')
    # Save a checkpoint
    if valid_f1 > valid_f1_prev:
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
            'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
                    os.path.join(output_dir, 'ner_bert_checkpoint.pt'))
        valid_f1_prev = valid_f1

evaluate(model, test_dataloader, batch_size, total_train_epochs-1, 'Test_set')

#%%
'''
Test_set prediction using the best epoch of NER_BERT model
'''
checkpoint = torch.load(output_dir+'/ner_bert_checkpoint.pt', map_location='cpu')
epoch = checkpoint['epoch']
valid_acc_prev = checkpoint['valid_acc']
valid_f1_prev = checkpoint['valid_f1']
model = BertForTokenClassification.from_pretrained(
    bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))
# if os.path.exists(output_dir+'/ner_bert_crf_checkpoint.pt'):
model.to(device)
print('Loaded the pretrain NER_BERT model, epoch:',checkpoint['epoch'],'valid acc:', 
        checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

model.to(device)
# evaluate(model, train_dataloader, batch_size, total_train_epochs-1, 'Train_set')
evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')


#%%
'''
#####  Use BertModel + CRF  #####
CRF is for transition and the maximum likelyhood estimate(MLE).
Bert is for latent label -> Emission of word embedding.
'''
print('*** Use BertModel + CRF ***')

def log_sum_exp_1vec(vec):  # shape(1,m)
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_mat(log_M, axis=-1):  # shape(n,m)
    return torch.max(log_M, axis)[0]+torch.log(torch.exp(log_M-torch.max(log_M, axis)[0][:, None]).sum(axis))

def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))


class BERT_CRF_NER(nn.Module):

    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, max_seq_length, batch_size, device):
        super(BERT_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        # self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device=device

        # use pretrainded BertModel 
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)
        # Maps the output of the bert into label space.
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)
        # self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)): 
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX 
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]
        
        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0
        
        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, input_ids, segment_ids, input_mask):
        '''
        sentances -> word embedding -> lstm -> MLP -> feats
        '''
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_all_encoded_layers=False)
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        return bert_feats

    def _score_sentence(self, feats, label_ids):
        ''' 
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size,self.num_labels,self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).to(device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t]*self.num_labels+label_ids[:, t-1]).view(-1,1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1,1)).view(-1,1)
        return score

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0
        
        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(bert_feats)
        # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        gold_score = self._score_sentence(bert_feats, label_ids)
        # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
        return torch.mean(forward_score - gold_score)

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask):
        # Get the emission scores from the BiLSTM
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)

        # Find the best path, given the features.
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids


start_label_id = conllProcessor.get_start_label_id()
stop_label_id = conllProcessor.get_stop_label_id()

bert_model = BertModel.from_pretrained(bert_model_scale)
model = BERT_CRF_NER(bert_model, start_label_id, stop_label_id, len(label_list), max_seq_length, batch_size, device)

#%%
if load_checkpoint and os.path.exists(output_dir+'/ner_bert_crf_checkpoint.pt'):
    checkpoint = torch.load(output_dir+'/ner_bert_crf_checkpoint.pt', map_location='cpu')
    start_epoch = checkpoint['epoch']+1
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    pretrained_dict=checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    print('Loaded the pretrain NER_BERT_CRF model, epoch:',checkpoint['epoch'],'valid acc:', 
            checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
else:
    start_epoch = 0
    valid_acc_prev = 0
    valid_f1_prev = 0

model.to(device)

# Prepare optimizer
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
        and not any(nd in n for nd in new_param)], 'weight_decay': weight_decay_finetune},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
        and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
    {'params': [p for n, p in param_optimizer if n in ('transitions','hidden2label.weight')] \
        , 'lr':lr0_crf_fc, 'weight_decay': weight_decay_crf_fc},
    {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
        , 'lr':lr0_crf_fc, 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate0)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    # print("***** Running prediction *****")
    model.eval()
    all_preds = []
    all_labels = []
    total=0
    correct=0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
            # _, predicted = torch.max(out_scores, -1)
            valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct/total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend:%.3f minutes for evaluation' \
        % (epoch_th, 100.*test_acc, 100.*precision, 100.*recall, 100.*f1, dataset_name,(end-start)/60.0))
    print('--------------------------------------------------------------')
    return test_acc, f1

#%%
# train procedure
global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)

train_start=time.time()
# for epoch in trange(start_epoch, total_train_epochs, desc="Epoch"):
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

        neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, input_mask, label_ids)

        if gradient_accumulation_steps > 1:
            neg_log_likelihood = neg_log_likelihood / gradient_accumulation_steps

        neg_log_likelihood.backward()

        tr_loss += neg_log_likelihood.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = learning_rate0 * warmup_linear(global_step_th/total_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
            
        print("Epoch:{}-{}/{}, Negative loglikelihood: {} ".format(epoch, step, len(train_dataloader), neg_log_likelihood.item()))
    
    print('--------------------------------------------------------------')
    print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss, (time.time() - train_start)/60.0))
    valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')

    # Save a checkpoint
    if valid_f1 > valid_f1_prev:
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
            'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
                    os.path.join(output_dir, 'ner_bert_crf_checkpoint.pt'))
        valid_f1_prev = valid_f1

evaluate(model, test_dataloader, batch_size, total_train_epochs-1, 'Test_set')


#%%
'''
Test_set prediction using the best epoch of NER_BERT_CRF model
'''
checkpoint = torch.load(output_dir+'/ner_bert_crf_checkpoint.pt', map_location='cpu')
epoch = checkpoint['epoch']
valid_acc_prev = checkpoint['valid_acc']
valid_f1_prev = checkpoint['valid_f1']
pretrained_dict=checkpoint['model_state']
net_state_dict = model.state_dict()
pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
net_state_dict.update(pretrained_dict_selected)
model.load_state_dict(net_state_dict)
print('Loaded the pretrain  NER_BERT_CRF  model, epoch:',checkpoint['epoch'],'valid acc:', 
      checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

model.to(device)
#evaluate(model, train_dataloader, batch_size, total_train_epochs-1, 'Train_set')
evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')
print('Total spend:',(time.time()-train_start)/60.0)


#%%
model.eval()
with torch.no_grad():
    demon_dataloader = data.DataLoader(dataset=test_dataset,
                                batch_size=10,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
    for batch in demon_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
        # _, predicted = torch.max(out_scores, -1)
        valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
        # valid_label_ids = torch.masked_select(label_ids, predict_mask)
        for i in range(10):
            print(predicted_label_seq_ids[i])
            print(label_ids[i])
            new_ids=predicted_label_seq_ids[i].cpu().numpy()[predict_mask[i].cpu().numpy()==1]
            print(list(map(lambda i: label_list[i], new_ids)))
            print(test_examples[i].labels)
        break
#%%
print(conllProcessor.get_label_map())
# print(test_examples[8].words)
# print(test_features[8].label_ids)
