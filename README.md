# BERT-NER
This is a named entity recognizer based on [BERT Model(pytorch-pretrained-BERT)](https://github.com/huggingface/pytorch-pretrained-BERT) and CRF.
## Requirements
- python 3.7
- pytorch 1.0.0
- pytorch-pretrained-bert 0.4.0
## Overview
The NER_BERT_CRF.py include 2 model:
- model 1:
  - This is just a pretrained BertForTokenClassification, For a comparision with my BERT-CRF model
- model 2:
  - A pretrained BERT with CRF model.
- data set
  - [CoNLL-2003](https://github.com/FuYanzhe2/Name-Entity-Recognition/tree/master/BERT-BiLSTM-CRF-NER/NERdata/ori)
### parameters
- NER_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', '[CLS]', '[SEP]', 'X']
- max_seq_length = 128
- batch_size = 32
- learning_rate = 2e-5
- total_train_epochs = 5
- do_lower_case = True
### Performance
- BertForTokenClassification
  - Accuracy on train data: 99.772 %
  - Accuracy of test data: 97.758 %
- Bert+CRF
  - Accuracy on train data: 99.940 %
  - Accuracy of test data: 97.906 % (98.75 % for total epochs = 8)
### Reference
[Bert paper](https://arxiv.org/abs/1810.04805)
[Bert with PyTorch implementation](https://github.com/huggingface/pytorch-pretrained-BERT)
[Bert-ner](https://github.com/ericput/bert-ner)
[CoNLL-2003 data set](https://github.com/FuYanzhe2/Name-Entity-Recognition/tree/master/BERT-BiLSTM-CRF-NER/NERdata)
