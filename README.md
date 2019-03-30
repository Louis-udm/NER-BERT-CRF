# NER implementation with BERT and CRF model
> Zhibin Lu

This is a named entity recognizer based on [BERT Model(pytorch-pretrained-BERT)](https://github.com/huggingface/pytorch-pretrained-BERT) and CRF.

Someone construct model with BERT, LSTM and CRF, like this [BERT-BiLSTM-CRF-NER](https://github.com/FuYanzhe2/Name-Entity-Recognition/tree/master/BERT-BiLSTM-CRF-NER), but in theory, the BERT mechanism has replaced the role of LSTM, so I think LSTM is redundant.

## Requirements
- python 3.6
- pytorch 1.0.0
- pytorch-pretrained-bert 0.4.0
## Overview
The NER_BERT_CRF.py include 2 model:
- model 1:
  - This is just a pretrained BertForTokenClassification, For a comparision with my BERT-CRF model
- model 2:
  - A pretrained BERT with CRF model.
- data set
  - [CoNLL-2003](https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/neuroner/data/conll2003/en)
### Parameters
- NER_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', '[CLS]', '[SEP]', 'X']
- max_seq_length = 128
- batch_size = 32
- learning_rate = 2e-5
- total_train_epochs = 5
- bert_model_type = 'bert-base-uncased'
- do_lower_case = True
### Performance
- [Bert paper](https://arxiv.org/abs/1810.04805)
  - F1-Score on valid data: 96.4 %
  - F1-Score on test data: 92.4 %
- BertForTokenClassification (epochs = 8)
  - Accuracy on valid data: 99.772 %
  - Accuracy on test data: 97.758 %
  - F1-Score on valid data: 
  - F1-Score on test data: 
- Bert+CRF (epochs = 8)
  - Accuracy on valid data: 99.940 %
  - Accuracy of test data: 97.906 % 
  - F1-Score on valid data: 
  - F1-Score on test data: 
### References
- [Bert paper](https://arxiv.org/abs/1810.04805)
- [Bert with PyTorch implementation](https://github.com/huggingface/pytorch-pretrained-BERT)
- [Bert-ner](https://github.com/ericput/bert-ner)
- [CoNLL-2003 data set](https://github.com/FuYanzhe2/Name-Entity-Recognition/tree/master/BERT-BiLSTM-CRF-NER/NERdata)
