import os
from transformers import BertTokenizer, BertForMaskedLM, BertModel, RobertaTokenizer, RobertaForMaskedLM, RobertaModel
import torch
from torch.utils.data import SequentialSampler, TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import logging


def init_logging_path(log_path,task_name,file_name):
    dir_log = os.path.join(log_path,f"{task_name}/{file_name}/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    return dir_log


def tokenize_mask(file_name, max_seq_len, bert_version="bert-large-uncased"):
    """
    Tokenize sentences for BERT Masked Model or Roberta Masked model
    :param file_name: path/noun.txt
    :param max_seq_len:
    :param bert_version: version of used model. e.g. roberta-large, bert-large-uncased. default = bert-large-uncased
    :return:
    list of token ids, input mask and indices of mention entity
    """
    if bert_version.split('-')[0]=='bert':
        tokenizer = BertTokenizer.from_pretrained(bert_version)
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        mask_token = "[MASK]"
    elif bert_version.split('-')[0] == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(bert_version)
        cls_token = "<s>"
        sep_token = "</s>"
        mask_token = "<mask>"
    token_ids = []
    input_mask = []
    indices = []
    word = os.path.basename(file_name).split('.')[0]
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            sent = line.strip().lower()
            left_seq, _, right_seq = sent.partition(str(word))
            tokens = [cls_token] + tokenizer.tokenize(left_seq)
            idx = len(tokens)
            tokens += [mask_token]
            tokens += tokenizer.tokenize(right_seq) + [sep_token]
            # print(tokens)
            if len(tokens) > max_seq_len:
                continue
            indices.append(idx)
            t_id = tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (max_seq_len - len(t_id))
            i_mask = [1] * len(t_id) + padding
            t_id += padding
            token_ids.append(t_id)
            input_mask.append(i_mask)
    return token_ids, input_mask, indices


def tokenize_nomask(file_name, max_seq_len, bert_version="bert-large-uncased"):
    """
    Tokenize sentences for BERT Model or Roberta Model
    :param file_name: path/noun.txt
    :param max_seq_len:
    :param bert_version: version of used model. e.g. roberta-large, bert-large-uncased. default = bert-large-uncased
    :return:
    list of token ids, input_mask and indices of mention entity, where each item in indices includes both start_idx and num of tokens of entity mention
    """
    if bert_version.split('-')[0] == 'bert':
        tokenizer = BertTokenizer.from_pretrained(bert_version)
        cls_token = "[CLS]"
        sep_token = "[SEP]"
    elif bert_version.split('-')[0] == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(bert_version)
        cls_token = "<s>"
        sep_token = "</s>"
    token_ids = []
    indices = []
    input_mask = []
    word = os.path.basename(file_name).split('.')[0]
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            sent = line.strip().lower()
            left_seq, _, right_seq = sent.partition(str(word))
            tokens = [cls_token] + tokenizer.tokenize(left_seq)
            word_tokens = tokenizer.tokenize(word)
            idx = [len(tokens), len(word_tokens)]
            tokens += word_tokens
            tokens += tokenizer.tokenize(right_seq) + [sep_token]

            # print(tokens)
            if len(tokens) > max_seq_len:
                continue
            indices.append(idx)
            t_id = tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (max_seq_len - len(t_id))
            i_mask = [1] * len(t_id) + padding
            t_id += padding
            token_ids.append(t_id)
            input_mask.append(i_mask)
    return token_ids, input_mask, indices


def extract_mv_mask(token_ids, input_mask, word_indices, batch_size, bert_version="bert-large-uncased", usegpu=False):
    """
        Extract mention vectors using Bert Masked Model or Roberta Masked Model
        :param token_ids: list (num_sent, max_seq_len), each row in the list corresponds to the token id list of a sentence
        :param word_indices: index list of target word (phrase), len = num_sent
        :param bert_version: version of used model. e.g. roberta-large, bert-large-uncased. default = bert-large-uncased
        :param batch_size
        :param usegpu
        :return:
        list of mention vectors
        """
    if bert_version.split('-')[0] == 'bert':
        mask_model = BertForMaskedLM.from_pretrained(bert_version, output_hidden_states=True)
    elif bert_version.split('-')[0] == 'roberta':
        mask_model = RobertaForMaskedLM.from_pretrained(bert_version, output_hidden_states=True)

    tokens_tensor = torch.tensor([ids for ids in token_ids], dtype=torch.long)
    input_mask_tensor = torch.tensor([im for im in input_mask], dtype=torch.long)
    indices = torch.tensor(word_indices)
    data = TensorDataset(tokens_tensor, input_mask_tensor, indices)

    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if usegpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            logging.info("use multiple gpu, n_gpu = " + str(n_gpu))
            mask_model = torch.nn.DataParallel(mask_model)
        mask_model.to('cuda')  # use GPU

    num_layers = mask_model.module.config.num_hidden_layers if isinstance(mask_model,
                                                                     torch.nn.DataParallel) else mask_model.config.num_hidden_layers
    hidden_dim = mask_model.module.config.hidden_size if isinstance(mask_model,
                                                               torch.nn.DataParallel) else mask_model.config.hidden_size
    results = torch.empty(num_layers, len(word_indices), hidden_dim)

    # if usegpu:
    #     results = results.to('cuda')
    start_idx = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        token_id, in_mask, index = batch
        if usegpu:
            token_id = token_id.to('cuda')  # GPU
            in_mask = in_mask.to('cuda')
            # index = index.to('cuda')

        with torch.no_grad():
            hidden_states = mask_model(token_id, attention_mask=in_mask)[1]  # all hidden-states(initial_embedding + all hidden layers), batch_size, sequence_length, dim

        # print('hidden states: ', len(hidden_states), hidden_states[0].shape)
        sent_ids = list(range(hidden_states[0].shape[0]))
        end_idx = start_idx + len(sent_ids)
        for i in range(1, len(hidden_states)):  # 0-layer is the initial embedding
            results[i-1][start_idx:end_idx] = hidden_states[i][sent_ids, index]
            torch.cuda.empty_cache()
        start_idx = end_idx

    return results


def extract_mv_mask_last(token_ids, input_mask, word_indices, batch_size, bert_version="bert-large-uncased", usegpu=False):
    """
        Extract mention vectors using Bert Masked Model or Roberta Masked Model
        :param token_ids: list (num_sent, max_seq_len), each row in the list corresponds to the token id list of a sentence
        :param word_indices: index list of target word (phrase), len = num_sent
        :param bert_version: version of used model. e.g. roberta-large, bert-large-uncased. default = bert-large-uncased
        :param batch_size
        :param usegpu
        :return:
        list of mention vectors
        """
    if bert_version.split('-')[0] == 'bert':
        mask_model = BertForMaskedLM.from_pretrained(bert_version, output_hidden_states=True)
    elif bert_version.split('-')[0] == 'roberta':
        mask_model = RobertaForMaskedLM.from_pretrained(bert_version, output_hidden_states=True)

    tokens_tensor = torch.tensor([ids for ids in token_ids], dtype=torch.long)
    input_mask_tensor = torch.tensor([im for im in input_mask], dtype=torch.long)
    indices = torch.tensor(word_indices)
    data = TensorDataset(tokens_tensor, input_mask_tensor, indices)

    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if usegpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            logging.info("use multiple gpu, n_gpu = " + str(n_gpu))
            mask_model = torch.nn.DataParallel(mask_model)
        mask_model.to('cuda')  # use GPU

    hidden_dim = mask_model.module.config.hidden_size if isinstance(mask_model,
                                                               torch.nn.DataParallel) else mask_model.config.hidden_size
    results = torch.empty(len(word_indices), hidden_dim)

    # if usegpu:
    #     results = results.to('cuda')
    start_idx = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        token_id, in_mask, index = batch
        if usegpu:
            token_id = token_id.to('cuda')  # GPU
            in_mask = in_mask.to('cuda')
            # index = index.to('cuda')

        with torch.no_grad():
            hidden_states = mask_model(token_id, attention_mask=in_mask)[1][-1]  # last hidden-states (batch_size, sequence_length, dim)

        # print(hidden_states.shape)
        sent_ids = list(range(hidden_states.shape[0]))
        end_idx = start_idx + len(sent_ids)
        results[start_idx:end_idx] = hidden_states[sent_ids, index]
        start_idx = end_idx

    return results


def extract_mv_nomask(token_ids, input_mask, word_indices, batch_size, bert_version="bert-large-uncased", usegpu=False):
    """
    Extract mention vectors using Bert Model or Roberta Model
    :param token_ids_list: list (num_sent, max_seq_len), each row in the list corresponds to the token id list of a sentence
    :param word_indices: index list of target word (phrase), len = num_sent, each item is a list [start_idx, num_tokens_word]
    :param bert_version: version of used model. e.g. roberta-large, bert-large-uncased. default = bert-large-uncased
    :param batch_size
    :param usegpu
    :return:
    list of mention vectors
    """
    if bert_version.split('-')[0] == 'bert':
        model = BertModel.from_pretrained(bert_version, output_hidden_states=True)#.get_input_embeddings()
    elif bert_version.split('-')[0] == 'roberta':
        model = RobertaModel.from_pretrained(bert_version, output_hidden_states=True)
    logging.info('model is Bert Model.')
    tokens_tensor = torch.tensor([ids for ids in token_ids], dtype=torch.long)
    input_mask_tensor = torch.tensor([im for im in input_mask], dtype=torch.long)
    indices = torch.tensor([id for id in word_indices], dtype=torch.int)
    data = TensorDataset(tokens_tensor, input_mask_tensor, indices)

    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if usegpu:
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model.to('cuda')  # use GPU

    num_layers = model.module.config.num_hidden_layers if isinstance(model, torch.nn.DataParallel) else model.config.num_hidden_layers
    hidden_dim = model.module.config.hidden_size if isinstance(model, torch.nn.DataParallel) else model.config.hidden_size
    results = torch.empty(num_layers, len(word_indices), hidden_dim)
    #if usegpu:
    #    results = results.to('cuda')
    start_idx = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        token_id, in_mask, index = batch
        if usegpu:
            token_id = token_id.to('cuda')  # GPU
            in_mask = in_mask.to('cuda')

        with torch.no_grad():
            hidden_states = model(token_id, attention_mask=in_mask)[2]  # all hidden states, 0 is initial embedding
        # print('hidden states: ', len(hidden_states), hidden_states[0].shape)
        end_idx = start_idx + hidden_states[0].shape[0]
        s_idx = index[:, 0]
        num = index[:, 1]
        for l in range(1, len(hidden_states)):
            avg_vec = torch.tensor([], dtype=torch.float)
            if usegpu:
                avg_vec = avg_vec.to('cuda')
            for ii in range(hidden_states[l].shape[0]):
                vec = hidden_states[l][ii, s_idx[ii]: s_idx[ii] + num[ii]]
                avg_vec = torch.cat((avg_vec, torch.mean(vec, dim=0).view(1, -1)))

            results[l-1][start_idx:end_idx] = avg_vec.cpu()
            torch.cuda.empty_cache()
        start_idx = end_idx
    return results


def write_out(filename, output):
    with open(filename, 'a+', encoding='utf-8') as f:
        f.write(output)

import csv
def write_csv(output_file, output):
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(output)



# import os
# # for ff in os.listdir("../sents"):
# #     token_ids, input_mask, indices = tokenize_mask(os.path.join("../sents", ff), 128, bert_version='roberta-base')
# #     # print(token_ids)
# #     re = extract_mv_mask_last(token_ids, input_mask, indices, 5, bert_version='roberta-base')
# #     print(ff, re.size())
    # break
