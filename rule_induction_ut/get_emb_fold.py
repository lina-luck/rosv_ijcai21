import os
import numpy as np
import re
import pandas as pd
import csv
from sklearn.decomposition import PCA

# def sep_by_uppercase(str):
#     pattern = "[A-Z]+"
#     new_str = re.sub(pattern, lambda x: "_" + x.group(0), str)
#     if str[0].islower():  # first character is lowercase
#         return new_str
#     else:
#         return new_str[1:]


def load_pretrained_embeddings(file):
    embeddings_dict = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], "float")
            embeddings_dict[word] = vector
    return embeddings_dict


# static_emb = load_pretrained_embeddings('dataset/static.txt')


def word_embedding(embeding_file, nodes_dict):
    nodes_keep = []
    embeddings = load_pretrained_embeddings(embeding_file)
    node_embeding = []
    for nod in nodes_dict:
        if nod in embeddings:
            node_embeding.append(embeddings[nod])
            nodes_keep.append(nod)
    return node_embeding, nodes_keep


# key = template name, value = id
def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d


def main():
    datasets = ['sumo'] #['wine', 'economy', 'olympics', 'transport']
    fold = 3

    embedding_files = ['avg_mv_mask.txt', 'avg_mv_nomask.txt', 'static.txt']
    for embedding_file in embedding_files:
        for dataset in datasets:
            for i in range(fold):
                path = 'dataset/' + dataset + '/' + str(fold) + '_fold' + '/'
                train_path = path + 'train/s' + str(i + 1)
                node_dict_file = train_path + '/nodes.dict'
                if 'nomask' in embedding_file:
                    feature_file = train_path + '/nomask_avg_mv_features.csv'
                elif 'static' in embedding_file:
                    feature_file = train_path + '/bert_static_features.csv'
                else:
                    feature_file = train_path + '/mask_avg_mv_features.csv'

                nodes_dict = _read_dictionary(node_dict_file)
                node_features, nodes_keep = word_embedding('dataset/' + embedding_file, nodes_dict)
                print(len(node_features[0]))
                f = open(feature_file, 'w', encoding='utf-8', newline='')
                writer = csv.writer(f)
                writer.writerow(node_features[0])
                writer.writerows(node_features)
                f.close()
                

                f = open(feature_file.replace('.csv', '_300.csv'), 'w', encoding='utf-8', newline='')
                pca = PCA(n_components=300, svd_solver='randomized')
                node_features = pca.fit_transform(np.array(node_features))
                print(node_features.shape[1])
                writer = csv.writer(f)
                writer.writerow(node_features[0])
                writer.writerows(node_features)
                f.close()
                
                f = open(train_path + '/nodes.dict', 'w', encoding='utf-8')
                i = 0
                for n in nodes_keep:
                    f.write(str(i) + '\t' + n + '\n')
                    i += 1
                f.close()

                break
main()
