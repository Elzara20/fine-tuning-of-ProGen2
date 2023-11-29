# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import os
import time
import random
import argparse
import re
import torch

from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, DefaultDataCollator, DataCollatorForLanguageModeling
import datasets
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import numpy as np
import matplotlib.pyplot as plt

from transformers.trainer_callback import PrinterCallback, ProgressCallback, EarlyStoppingCallback, DefaultFlowCallback
from transformers.integrations import TensorBoardCallback

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


########################################################################
# util


class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic



########################################################################
# model


def create_model(ckpt, fp16=True):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


########################################################################
# sample


def sample(device, model, tokenizer, context, max_length, num_return_sequences, top_p, temp, pad_token_id):
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
        tokens_batch = model.generate(input_ids, do_sample=True, temperature=temp, max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
        as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]
        return tokenizer.decode_batch(as_lists(tokens_batch))


def truncate(sample, terminals):
    pos = []
    for terminal in terminals:
        find_pos = sample.find(terminal, 1)
        if find_pos != -1:
            pos.append(find_pos)
    if len(pos) > 0:
        return sample[:(min(pos)+1)]
    else:
        return sample


def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)




########################################################################
# main


def main():
    # (0) constants

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B
    
    # (1) params

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=models, default='progen2-large')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', default=True, type=lambda x: (str(x).lower() == 'true'))

    parser.add_argument('--link-data', type=str, default='./seqdump.txt')
    parser.add_argument('--comet-api_key', type=str, default='None')
    parser.add_argument('--comet-project', type=str, default='None')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=6e-5)
    parser.add_argument('--strategy', type=str, default='steps')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dir-finetune', type=str, default='./')
    parser.add_argument('--dir-output', type=str, default='./')
    parser.add_argument('--dir-logging', type=str, default='./')
    
    args = parser.parse_args()

    #check
    if args.strategy not in ['steps', 'epoch', 'no']:
      raise ValueError(f"strategy must be 'steps', 'epoch' or 'no', received value = {args.strategy}")

    # (1.1) preparation of data for transfer learning
    dataset = open(args.link_data, "r").readlines()
    # разделение на сами последовательности и их id
    dataset_names, dataset_seq = [], []
    seq=''
    for i in range(len(dataset)):
      res=re.findall(">.+", dataset[i])
      
      if len(res)!=0:
        dataset_seq.append(seq)
        dataset_names.append(dataset[i])
        seq=''
      else:
        seq+=dataset[i]
    dataset_seq.append(seq)
    # создание единой последовательности, так как после blast в последовательностях есть \n
    print(dataset_seq[:3])
    count_empty=0
    dataset_seq_new=[]
    for i in dataset_seq:
      if len(i)!=0:
        dataset_seq_new.append(i.replace("\n", ""))
      else:
        count_empty+=1
    # (2) preamble

    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    if not torch.cuda.is_available():
        print('falling back to cpu')
        args.device = 'cpu'

    device = torch.device(args.device)
    ckpt = f'./checkpoints/{args.model}'

    if device.type == 'cpu':
        print('falling back to fp32')
        args.fp16 = False

    # (3) load

    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt, fp16=args.fp16).to(device)
    

    with print_time('loading tokenizer'):
        tokenizer = create_tokenizer_custom(file='tokenizer.json') 
    
    # (4) sanity -> готовность к работе (всегда срабатывает )
    # (5) sample




    # (6) prepair training data for model
    # (6.1) split data into training and validation dataset
    # dataset_seq_new = dataset_seq_new[:100] # check
    data_seq_train = dataset_seq_new[:int(len(dataset_seq_new)*0.8)]
    data_seq_val=dataset_seq_new[int(len(dataset_seq_new)*0.8):]
    data_name_train =dataset_names[:int(len(dataset_names)*0.8)]
    data_name_val=dataset_names[int(len(dataset_names)*0.8):]
    # (6.2) tokenize data
    data_seq_train_new=[]
    data_seq_val_new=[]
    for i in data_seq_train:
      data_seq_train_new.append(torch.tensor(tokenizer.encode(i).ids).to(device))
    for i in data_seq_val:
      data_seq_val_new.append(torch.tensor(tokenizer.encode(i).ids).to(device))
    # (6.3) formation datasets
    train_dataset1 = datasets.Dataset.from_dict({'input_ids': data_seq_train_new}) #, 'labels': data_seq_train_new})
    val_dataset1 = datasets.Dataset.from_dict({'input_ids': data_seq_val_new}) #, 'labels': data_seq_val_new})
    all_dataset = datasets.DatasetDict({'train': train_dataset1, 'val':val_dataset1})
    # (6.4) data collator -> предобработка данных (такое как одинаковое кол-во длины последовательности) в батчи для более эффективной работы модели
    class MyDataCollator(DefaultDataCollator): #DataCollatorForLanguageModeling
      def __init__(self, tokenizer):
          super(MyDataCollator, self).__init__(tokenizer)
          self.tokenizer = tokenizer
          self.tokenizer.pad_token_id = tokenizer.encode('<|pad|>').ids[0]
      def __call__(self, seq):
          seq[0]['input_ids'] = torch.tensor(seq[0]['input_ids'])
          return seq[0] #torch.tensor(seq[0]['input_ids'])
      


    data_collator = MyDataCollator(tokenizer)

    # (6.5) training arguments
    comet_flag=0
    
   
    training_args = TrainingArguments(
      output_dir=args.dir_finetune, #все файлы
      logging_dir=args.dir_logging,
      overwrite_output_dir=True,
      num_train_epochs=args.epoch,
      save_steps=10, 
      learning_rate=args.learning_rate,
      save_total_limit=2, # получим количество папок в chaeckpoint - лучшая и предлучшая модель,
      logging_steps=10, # как часто training logs будут выводится в консоль
      load_best_model_at_end = True,
      evaluation_strategy = args.strategy,
      save_strategy = args.strategy,
      label_names=['input_ids'],
      weight_decay=args.weight_decay,
      # report_to="all"
    )
    
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
    


    trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=train_dataset1,
      eval_dataset=val_dataset1,
      callbacks=[PrinterCallback(), ProgressCallback(), DefaultFlowCallback(), TensorBoardCallback, early_stopping_callback],
    
    )
    
    trainer.train() 
    trainer.save_model(args.dir_finetune)
    
    
    # значение потерь при тренировке и валидации
    with open(f'{args.dir_output}/trainer_log_history.txt', 'w') as f:
      print(trainer.state.log_history, file=f)

    train_loss=[]
    val_loss = []
    for i in trainer.state.log_history:
      if 'loss' in i:
        train_loss.append(i['loss'])
      if 'eval_loss' in i:
        val_loss.append(i["eval_loss"])
    
    plt.figure(figsize=(15, 10))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{args.dir_output}/train_val_loss.png')
    plt.savefig(f'{args.dir_output}/train_val_loss.pdf')
    plt.show()

    trainer.evaluate() 
    
            


if __name__ == '__main__':
    main()
    print('done.')
