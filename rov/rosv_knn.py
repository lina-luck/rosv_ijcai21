import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
import getopt
from src.rosv_util import *


def main(argv):
    log_dir = os.path.join(os.getcwd(), "log/")
    logfile = ''

    max_sent_len = '64'
    maxk = 300
    use_mask = False
    data_path = ''
    mv_path = ''
    out_path = ''

    try:
        opts, args = getopt.getopt(argv, "hl:d:v:o:k:m", ["lfile=", "ddir=", "vdir=", "odir=", "maxk="])
    except getopt.GetoptError:
        print('rosv_knn.py -l <logfile> -d <data_path> -v <men_vec_path> -o <output_path> -m <use mask>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('rosv_knn.py -l <logfile> -d <data_path> -v <men_vec_path> -o <output_path> -m <use mask>')
            sys.exit()
        elif opt in ("-l", "--lfile"):
            logfile = arg
        elif opt in ("-d", "--ddir"):
            data_path = arg
        elif opt in ("-v", "--vdir"):
            mv_path = arg
        elif opt in ("-o", "--odir"):
            out_path = arg
        elif opt in ("-k", "--maxk"):
            maxk = int(arg)
        elif opt == "-m":
            use_mask = True

    log_file_path = init_logging_path(log_dir, 'knn', logfile)
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        )

    logging.info("start logging: " + log_file_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError("data path not exists")

    dataset = os.path.normpath(data_path).split('/')[-1]
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    logging.info("loading samples...")

    if dataset in ['cslb', 'mcrae', 'wordnet', 'babelnet', 'morrow', 'abstract']:
        train = load_nouns(os.path.join(data_path, 'train_inst.txt'))
        valid = load_nouns(os.path.join(data_path, 'val_inst.txt'))
        test = load_nouns(os.path.join(data_path, 'test_inst.txt'))
    elif dataset in ['glasgow', 'mt40k', 'anew', 'morrowA', 'morrowB']:
        train, _ = load_data(os.path.join(data_path, 'train.txt'))
        test, _ = load_data(os.path.join(data_path, 'test.txt'))
        valid = []

    if use_mask:
        out_file = os.path.join(out_path, 'knn_mask_large_' + dataset + '_' + str(maxk) + '.txt')
    else:
        out_file = os.path.join(out_path, 'knn_nomask_large_' + dataset + '_' + str(maxk) + '.txt')
    knn_train_data(train, mv_path, maxk, out_file)

    knn_eval_data(train, valid + test, mv_path, maxk, out_file)

    logging.info("Done.")


if __name__ == '__main__':
    main(sys.argv[1:])
