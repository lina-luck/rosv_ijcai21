import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
import getopt
import logging
from src.utils import *
import faiss
from faiss import normalize_L2
from sklearn.metrics import silhouette_score
import random


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


out_idx_path = 'data/neighbors/'
out_mv_path = 'data/avg_mv_remain/'
make_path(out_idx_path)
make_path(out_mv_path)


def write_out(file_name, output):
    with open(file_name, 'a+', encoding='utf-8') as f:
        f.writelines(output)


def get_centroid(men_vec, method='mean'):
    logging.info("Compute clustering centroid")
    men_vec = np.array(men_vec).astype('float32')
    dim = men_vec.shape[1]
    mean = None

    if method == 'mean':
        mean = np.mean(men_vec, axis=0)
    elif method == 'clustering':
        best_score = -1
        max_ncentriod = 10 #max(men_vec.shape[0] // 10, 3)
        for ncentriods in range(2, max_ncentriod):
            kmeans = faiss.Kmeans(dim, ncentriods, niter=500)
            kmeans.train(men_vec)
            _, pred_label = kmeans.index.search(men_vec, 1)
            pred_label = pred_label.reshape(-1)
            silhouette_avg = silhouette_score(men_vec, pred_label)
            #print(ncentriods, silhouette_avg)
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                max_cluster_idx = np.argmax(np.bincount(pred_label))
                mean = kmeans.centroids[max_cluster_idx]
    mean = mean.reshape(1, -1).astype('float32')
    return mean


def sort_mv(men_vec, mean, use_cosine=False):
    logging.info("Sort the mention vectors for each sample based on the distance (close to far) to centroid and write out")
    men_vec = np.array(men_vec).astype('float32')
    dim = men_vec.shape[1]
    if use_cosine:
        normalize_L2(men_vec)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(men_vec)
    logging.info("Total number of index is " + str(index.ntotal))

    _, I = index.search(mean, men_vec.shape[0])   # sorted index from close to far
    return I


def load_data(samples, mv_path, start_i=0):
    logging.info("load in mention vectors for samples")
    men_vec = []
    nouns_idx = {}
    i = start_i
    for noun in samples:
        vec = load_bert_vectors(mv_path, noun)
        men_vec.extend(vec)
        nouns_idx[noun] = list(range(i, i + len(vec)))
        i += len(vec)
    logging.info("Total number of training vector is " + str(len(men_vec)))
    return men_vec, nouns_idx


def main(argv):
    log_dir = os.path.join(os.getcwd(), "log/")
    logfile = ''

    max_sent_len = '64'
    cluster = 'mean'
    use_mask = False
    data_path = ''
    mv_path = ''
    candidate_k = [5, 10, 20, 50, 80]

    try:
        opts, args = getopt.getopt(argv, "hl:d:v:m:c:", ["lfile=", "ddir=", "vdir=", "mask=", "cluster="])
    except getopt.GetoptError:
        print('filtering_baseline1.py -l <logfile> -d <data_path> -v <men_vec_path> -m <use mask> -c <cluster method>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('filtering_baseline1.py -l <logfile> -d <data_path> -v <men_vec_path> -m <use mask> -c <cluster method>')
            sys.exit()
        elif opt in ("-l", "--lfile"):
            logfile = arg
        elif opt in ("-d", "--ddir"):
            data_path = arg
        elif opt in ("-v", "--vdir"):
            mv_path = arg
        elif opt in ("-m", "--mask"):
            use_mask = bool(arg)
        elif opt in ("-c", "--cluster"):
            cluster = arg
    print(use_mask, cluster)

    log_file_path = init_logging_path(log_dir, 'fb1', logfile)
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        )

    logging.info("start logging: " + log_file_path)

    dataset = os.path.normpath(data_path).split('/')[-1]

    if cluster == 'mean':
        tag = 'fb1'
    elif cluster == 'clustering':
        tag = 'fb2'
    else:
        print("wrong parameter for cluster")
        exit()

    logging.info("loading samples...")

    if dataset in ['abstract', 'cslb', 'morrow', 'wordnet', 'babelnet']:
        train = load_nouns(os.path.join(data_path, 'train_inst.txt'))
        valid = load_nouns(os.path.join(data_path, 'val_inst.txt'))
        test = load_nouns(os.path.join(data_path, 'test_inst.txt'))
    elif dataset in ['anew', 'glasgow']:
        train, _ = load_data(os.path.join(data_path, 'train.txt'))
        test, _ = load_data(os.path.join(data_path, 'test.txt'))
        valid = []

    men_vec, nouns_idx = load_data(train, mv_path)
    start_i = len(men_vec)
    mean = get_centroid(men_vec, cluster)
    men_vec1, nouns_idx1 = load_data(valid + test, mv_path, start_i)
    men_vec = np.concatenate((men_vec, np.array(men_vec1).astype("float32")))
    nouns_idx.update(nouns_idx1)
    del men_vec1
    del nouns_idx1
    sorted_idx = sort_mv(men_vec, mean)[0]

    avg_vec = dict()
    logging.info("Compute filtered average mention vectors for each sample")
    for k in candidate_k:
        keep_num = int(men_vec.shape[0] * (k/100))
        topk_idx = sorted_idx[:keep_num]

        for nn in nouns_idx:
            keep_id_n = [i for i in nouns_idx[nn] if i in topk_idx]
            if len(keep_id_n) == 0:
                keep_id_n = list(random.sample(nouns_idx[nn], 1))
            avg_vec[nn] = np.mean(np.array(men_vec)[keep_id_n], axis=0)

        if use_mask:
            mv_file = os.path.join(out_mv_path, tag + '_avg_mv_mask_large_' + dataset + '_' +
                                        str(k) + '.txt')
        else:
            mv_file = os.path.join(out_mv_path, tag + '_avg_mv_nomask_large_' + dataset + '_' +
                                            str(k) + '.txt')

        with open(mv_file, 'w', encoding='utf-8') as f:
            for key in avg_vec:
                f.write(' '.join([key] + [str(v) for v in avg_vec[key]]) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
