nodes_dict = dict()
f = open('nodes2keys.txt', 'r')
for line in f.readlines():
    nod, noun = line.strip().split('\t')
    nodes_dict[nod] = noun
f.close()


def load_pretrained_embeddings(file):
    embeddings_dict = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            embeddings_dict[word] = line
    return embeddings_dict

import os

out_mv_path = 'men_vec/mask_large_64_500/'
out_no_path = 'men_vec/nomask_large_64_500/'
if not os.path.exists(out_mv_path):
    os.makedirs(out_mv_path)

if not os.path.exists(out_no_path):
    os.makedirs(out_no_path)

import csv
def load_bert_vectors(path, noun):
    data = []
    with open(os.path.join(path, noun +'.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            vec = [float(l) for l in line]
            if vec not in data:
                data.append(vec)
    return data

mv_path1 = '/home/cril/lina/emnlp2020/data/embeddings/mask_large_men_vec_64/'
no_mv_path1 = '/home/cril/lina/emnlp2020/data/embeddings/nomask_large_men_vec_64/'
mv_path2 = '/home_stockage/cril/bouraoui_group/wordnet_nouns/wn_mv_64_500/wn_mask_large_mention_vector_64/'
no_mv_path2 = '/home_stockage/cril/bouraoui_group/wordnet_nouns/wn_mv_64_500/wn_nomask_large_mention_vector_64/'

mask_emb = load_pretrained_embeddings('/home/cril/lina/unary_classify/dataset/avg_mv_mask.txt')
nomask_emb = load_pretrained_embeddings('/home/cril/lina/unary_classify/dataset/avg_mv_nomask.txt')
static_emb = load_pretrained_embeddings('/home/cril/lina/unary_classify/dataset/static.txt')
avg_mask = ''
avg_nomask = ''
static = ''
for nod in nodes_dict:
    nn = nodes_dict[nod]
    if nn + '.csv' in os.listdir(out_no_path) or (nn + '.csv' not in os.listdir(mv_path1) and nn + '.csv' not in os.listdir(mv_path2)):
        continue
    if nn + '.csv' not in os.listdir(out_no_path):
        if nn + '.csv' in os.listdir(no_mv_path1):
            #os.popen('cp ' + mv_path1 + nn + '.csv ' + out_mv_path)
            os.popen('cp ' + no_mv_path1 + nn + '.csv ' + out_no_path)
        elif nn + '.csv' in os.listdir(no_mv_path2):
            #os.popen('cp ' + mv_path2 + nn + '.csv ' + out_mv_path)
            os.popen('cp ' + no_mv_path2 + nn + '.csv ' + out_no_path)
    avg_mask += mask_emb[nod]
    avg_nomask += nomask_emb[nod]
    static += static_emb[nod]
'''
f = open('dataset/avg_mv_mask.txt', 'a+')
f.write(avg_mask)
f.close()

f = open('dataset/avg_mv_nomask.txt', 'a+')
f.write(avg_nomask)
f.close()

f = open('dataset/static.txt', 'a+')
f.write(static)
f.close()



'''
