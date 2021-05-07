import os
import numpy as np
import re
import pandas as pd
import csv

def sep_by_uppercase(str):
    pattern = "[A-Z]+"
    new_str = re.sub(pattern, lambda x: "_" + x.group(0), str)
    if str[0].islower():  # first character is lowercase
        return new_str
    else:
        return new_str[1:]


def load_pretrained_embeddings(file):
    embeddings_dict = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            #print(line, len(values))
            word = values[0]
            vector = np.asarray(values[1:], "float")
            embeddings_dict[word] = vector
    return embeddings_dict


def word_embedding(embeding_file, nodes_dict):
    embeddings = load_pretrained_embeddings(embeding_file)
    node_embeding = np.zeros((len(nodes_dict), 1024))  # default value is 0
    for nod in nodes_dict:
        if nod in embeddings:
            node_embeding[nodes_dict[nod]] = embeddings[nod]
        else:
            print(nod, ' not in embedding')
    return node_embeding


# key = template name, value = id
def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d


def main():
    datasets = ['wine', 'economy', 'olympics', 'transport', 'sumo']
    for data in datasets:
        if data == 'sumo':
            fold = 3
        else:
            fold = 10

        for k in [5, 10, 20, 50, 100]:
            embedding_file = os.path.join('avg_mv_remain/nomask_large_' + data + '_' + str(k) + '.txt')
            for i in range(fold):
                path = 'dataset/' + data + '/' + str(fold) + '_fold' + '/'
                train_path = path + 'train/s' + str(i + 1)
                node_dict_file = train_path + '/nodes.dict'
                if 'nomask' in embedding_file:
                    feature_file = train_path + '/rosv_nomask_avg_mv_features_' + str(k) + '.csv'
                else:
                    feature_file = train_path + '/rosv_mask_avg_mv_features_' + str(k) +'.csv'
                print(data, k, i+1)
                nodes_dict = _read_dictionary(node_dict_file)
                node_features = word_embedding(embedding_file, nodes_dict)
                f = open(feature_file, 'w', encoding='utf-8', newline='')
                writer = csv.writer(f)
                writer.writerow(node_features[0])
                writer.writerows(node_features)
                f.close()
                if data == 'sumo':
                    break

main()
