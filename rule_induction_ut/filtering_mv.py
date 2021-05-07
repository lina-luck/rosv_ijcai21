import os
from rosv_util import *

mv_path = 'men_vec/nomask_large_64_500/'
if not os.path.exists('knn'):
    os.mkdir('knn')

out_mv_path = os.path.join('avg_mv_remain')
if not os.path.exists(out_mv_path):
    os.makedirs(out_mv_path)


# key = template name, value = id
def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d


org_vec = dict()
for ff in os.listdir(mv_path):
    n = ff.split('.')[0]
    org_vec[n] = np.array(load_bert_vectors(mv_path, n))


candidate_k = [5, 10, 20, 50, 100]
datasets = ['wine', 'economy', 'olympics', 'transport', 'sumo']
for data in datasets:
    print(data)
    nodes_dict = dict()
    if data == 'sumo':
        fold = '3_fold'
        node_dict_file = 'dataset/sumo/node2noun.txt'
        f = open(node_dict_file, 'r')
        for line in f.readlines():
            nod, noun = line.strip().split('\t')[1:]
            nodes_dict[nod] = noun
        f.close()
    else:
        fold = '10_fold'
        node_dict_file = 'nodes2keys.txt'
        f = open(node_dict_file, 'r')
        for line in f.readlines():
            nod, noun = line.strip().split('\t')
            nodes_dict[nod] = noun
        f.close()

    node_file = os.path.join('dataset', data, fold, 'train/s1/nodes.dict')
    nodes = _read_dictionary(node_file)
    nouns = [nodes_dict[nod] for nod in nodes]
    knn_file = os.path.join('knn', 'nomask_' + data + '_300.txt')
    knn_train_data(set(nouns), mv_path, 300, knn_file)
    knn = load_neighbors(knn_file, 300)
    for k in candidate_k:
        remain_noun_id = filter_strategy_rosv(knn[:, :k + 1])
        avg_vec = dict()
        for nod in nodes:
            noun = nodes_dict[nod]
            if noun in remain_noun_id:
                avg_vec[nod] = np.mean(org_vec[noun][remain_noun_id[noun]], axis=0)
            else:
                print(k, noun)
                avg_vec[nod] = np.mean(org_vec[noun], axis=0)

        mv_file = os.path.join(out_mv_path, 'nomask_large_' + data + '_' + str(k) + '.txt')
        with open(mv_file, 'w', encoding='utf-8') as f:
            for key in avg_vec:
                f.write(' '.join([key] + [str(v) for v in avg_vec[key]]) + '\n')










