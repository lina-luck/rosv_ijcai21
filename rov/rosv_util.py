import faiss
from faiss import normalize_L2
import logging
import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
from src.utils import *
import copy
from nltk.corpus import wordnet as wn


def load_neighbors(file_name, max_k):
    knn = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            noun, nn = line.strip().split('\t')
            knn.append([noun] + nn.split(',')[:max_k])
    return np.array(knn)


def write_out(file_name, output):
    with open(file_name, 'a+', encoding='utf-8') as f:
        f.writelines(output)


def knn_train_data(train_samples, mv_path, max_k, out_file, chunk_size=10000, use_cosine=False):
    logging.info("Start processing training samples")
    men_vec = []
    nouns = []
    for noun in train_samples:
        vec = load_bert_vectors(mv_path, noun)
        men_vec.extend(vec)
        nouns.extend([noun] * len(vec))
    logging.info("Total number of training vector is " + str(len(nouns)))

    nouns = np.array(nouns)
    men_vec = np.array(men_vec).astype('float32')
    dim = men_vec.shape[1]
    if use_cosine:
        normalize_L2(men_vec)
        index = faiss.IndexFlatIP(dim)
    else:
        normalize_L2(men_vec)
        index = faiss.IndexFlatL2(dim)
    #index = faiss.index_cpu_to_all_gpus(index)
    index.add(men_vec)

    logging.info("Total number of index is " + str(index.ntotal))

    knn = ''
    s = 0
    n_i = 0
    times = men_vec.shape[0] // chunk_size + 1
    for tt in range(times):
        if tt == times - 1:
            e = s + men_vec.shape[0] % chunk_size
        else:
            e = s + chunk_size
        _, I = index.search(men_vec[s:e], max_k + 1)

        for i in I:
            noun_ii = np.where(i == n_i)[0]
            if noun_ii.shape[0] == 1:
                knn += nouns[n_i] + '\t' + ','.join(nouns[np.delete(i, noun_ii[0])]) + '\n'
            else:
                knn += nouns[n_i] + '\t' + ','.join(nouns[i[:max_k]]) + '\n'
            n_i += 1
        s = e
        logging.info(str((tt + 1) * chunk_size / nouns.shape[0]) + " % training vectors processed.")

    write_out(out_file, knn)
    logging.info("knn for training samples were wrote out. ")


def knn_eval_data(train_samples, eval_samples, mv_path, max_k, out_file, use_cosine=False):
    logging.info("Start processing evaluation samples")
    men_vec = []
    nouns = []
    for noun in train_samples:
        vec = load_bert_vectors(mv_path, noun)
        men_vec.extend(vec)
        nouns.extend([noun] * len(vec))
    logging.info("Total number of training vector is " + str(len(nouns)))

    nouns = np.array(nouns)
    men_vec = np.array(men_vec)
    dim = men_vec.shape[1]

    knn = ''
    cnt = 0
    for noun in eval_samples:
        cnt += 1
        if cnt % 1000 == 0:
            logging.info(str(round(cnt / len(eval_samples) * 100, 2)) + " % nouns processed.")
        vec = load_bert_vectors(mv_path, noun)
        nouns2 = copy.deepcopy(nouns)
        nouns2 = np.append(nouns2, np.array([noun] * len(vec)))

        vec = np.vstack((men_vec, np.array(vec))).astype('float32')
        if use_cosine:
            normalize_L2(vec)
            index = faiss.IndexFlatIP(dim)
        else:
            normalize_L2(vec)
            index = faiss.IndexFlatL2(dim)
        #index = faiss.index_cpu_to_all_gpus(index)
        index.add(vec)

        n_j = men_vec.shape[0]
        _, I = index.search(vec[men_vec.shape[0]:], max_k + 1)
        for i in I:
            noun_ii = np.where(i == n_j)[0]
            if noun_ii.shape[0] == 1:
                knn += noun + '\t' + ','.join(nouns2[np.delete(i, noun_ii[0])]) + '\n'
            else:
                knn += noun + '\t' + ','.join(nouns2[i[:max_k]]) + '\n'
            n_j += 1

    write_out(out_file, knn)
    logging.info("knn for evaluation samples were wrote out. ")


def load_remain_noun_id(file):
    remain_noun_id = dict()
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = line.strip().split()
            noun = tmp[0]
            ids = [int(i) for i in tmp[1:]]
            remain_noun_id[noun] = ids
    return remain_noun_id


def filtered_sample(embedding_dict, samples):
    new_samples = []
    for noun in samples:
        if noun in embedding_dict:
            new_samples.append(noun)
    return new_samples


def filter_strategy_rosv(knn):
    nouns = knn[:, 0]
    noun_start_id = {}
    old_noun = ''
    for i in range(nouns.shape[0]):
        cur_noun = nouns[i]
        if old_noun == '' or not cur_noun == old_noun:
            noun_start_id[cur_noun] = i
            old_noun = cur_noun

    k = knn.shape[1] - 1
    target = nouns.repeat(k).reshape((-1, k))
    score = (knn[:, 1:] == target).astype(np.int).sum(axis=1)
    remain_idx = np.where(score < k)[0]
    remain_noun_id = {}
    for id in remain_idx:
        noun = nouns[id]
        if noun not in remain_noun_id:
            remain_noun_id[noun] = [id-noun_start_id[noun]]
        else:
            remain_noun_id[noun].append(id-noun_start_id[noun])
    return remain_noun_id
