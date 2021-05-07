import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
import getopt
from src.rosv_util import *
import random


def main(argv):
    log_dir = os.path.join(os.getcwd(), "log/")
    logfile = ''
    candidate_k = [5, 10, 20, 50, 100, 110, 115]
    max_sent_len = '64'
    use_mask = False
    knn_path = ''
    mv_path = ''
    dataset = ''
    maxk = 250

    try:
        opts, args = getopt.getopt(argv, "hd:n:l:v:m", ["data=", "nndir=", "lfile=", "vdir"])
    except getopt.GetoptError:
        print(
            "rosv_avg_mv.py -d <dataset> -n <knn_path> -m <use_mask> -l <logfile> -v <men_vec_path>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("rosv_avg_mv.py -d <dataset> -n <knn_path> -m <use_mask> -l <logfile> -v <men_vec_path>")
            sys.exit()
        elif opt in ("-d", "--data"):
            dataset = arg
        elif opt in ("-n", "--nndir"):
            knn_path = arg
        elif opt in ("-l", "--lfile"):
            logfile = arg
        elif opt in ("-v", "--vdir"):
            mv_path = arg
        elif opt == "-m":
            use_mask = True

    log_file_path = init_logging_path(log_dir, 'rosv_avg_mv', logfile)
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        )

    logging.info("start logging: " + log_file_path)

    if not os.path.exists(knn_path):
        raise FileNotFoundError("knn path not exists")

    out_mv_path = os.path.join(os.path.abspath(project_path), 'data', 'avg_mv_remain')
    if not os.path.exists(out_mv_path):
        os.makedirs(out_mv_path)

    logging.info("dataset: " + dataset)
    logging.info("avgerage mention vectors after filtering are stored under " + out_mv_path)

    nouns = load_nouns(os.path.join(os.path.abspath(project_path), 'data/dataset_nouns', dataset + '.txt'))

    logging.info("loading original vectors")
    org_vec = dict()
    for n in nouns:
        vec = load_bert_vectors(mv_path, n)
        org_vec[n] = np.array(vec)

    if use_mask:
        knn_file = os.path.join(knn_path, 'knn_mask_large_' + dataset + '_' + str(maxk) + '.txt')
    else:
        knn_file = os.path.join(knn_path, 'knn_nomask_large_' + dataset + '_' + str(maxk) + '.txt')

    knn = load_neighbors(knn_file, max(candidate_k))

    for k in candidate_k:
        logging.info("filtering for k = " + str(k))
        remain_noun_id = filter_strategy_rosv(knn[:, :k+1])

        logging.info("start computing avgerage mention vectors")
        avg_vec = dict()
        for n in nouns:
            if n in remain_noun_id:
                avg_vec[n] = np.mean(org_vec[n][remain_noun_id[n]], axis=0)
            else:
                logging.info(n + ' is filtered, replaced by a randomly selected vector')
                avg_vec[n] = org_vec[n][random.sample(list(range(org_vec[n].shape[0])), 1)[0]]

        if use_mask:
            mv_file = os.path.join(out_mv_path, 'rosv_avg_mv_mask_large_' + dataset + '_' + str(k) + '.txt')
        else:
            mv_file = os.path.join(out_mv_path, 'rosv_avg_mv_nomask_large_' + dataset + '_' + str(k) + '.txt')

        with open(mv_file, 'w', encoding='utf-8') as f:
            for key in avg_vec:
                f.write(' '.join([key] + [str(v) for v in avg_vec[key]]) + '\n')
        logging.info("avgerage mention vectors after filtering stored.")


if __name__ == '__main__':
    main(sys.argv[1:])
