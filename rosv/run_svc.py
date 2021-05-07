import getopt
import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
import numpy as np
from src.utils import load_prop_instances, load_pretrained_embeddings, word_embedding, write_csv, write_txt, init_logging_path
from src.svm import train_svc, test_svc
import logging


def main(argv):
    log_dir = os.path.join(os.getcwd(), "log/")
    logfile = ''

    vector_type = 'static'
    max_sent_len = '64'
    use_mask = False
    data_path = ''
    embed_path = ''
    kernel = ''

    try:
        opts, args = getopt.getopt(argv, "hd:v:k:e:m", ["data_dir=", "vector_type=", "kernel=", "emb_dir="])
    except getopt.GetoptError:
        print("run_svc.py -d <data_path> -v <vector_type> -m <use_mask> -k <kernel> -e <emb_path>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("python run_svc.py -d <data_path> -v <vector_type> -m <use_mask> -k <kernel> -e <emb_path>")
            sys.exit()
        elif opt in ("-d", "--data_dir"):
            data_path = arg
        elif opt in ("-v", "--vector_type"):
            vector_type = arg
        elif opt in ("-k", "--kernel"):
            kernel = arg
        elif opt in ("-e", "--emb_dir"):
            embed_path = arg
        elif opt == "-m":
            use_mask = True

    if not os.path.exists(data_path):
        raise FileNotFoundError("data path not exists")
    dataset = os.path.normpath(data_path).split('/')[-1]

    result_path = os.path.join(os.path.abspath(project_path), 'result', dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    log_file_path = init_logging_path(log_dir, 'extract_men_vec', logfile)
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info("start logging: " + log_file_path)
    logging.info("vector type is :" + vector_type)
    logging.info("dataset is " + dataset)
    logging.info("loading property and corresponding instances")

    pos_train = load_prop_instances(os.path.join(data_path, f'pos_train_data.txt'))
    pos_test = load_prop_instances(os.path.join(data_path, f"pos_test_data.txt"))
    neg_test = load_prop_instances(os.path.join(data_path, f"neg_test_data.txt"))
    pos_valid = load_prop_instances(os.path.join(data_path, f"pos_valid_data.txt"))
    neg_valid = load_prop_instances(os.path.join(data_path, f"neg_valid_data.txt"))
    if vector_type in ('glove', 'word2vec'):
        neg_train = load_prop_instances(os.path.join(data_path, f"neg_train_data.txt"))
    else:
        neg_train = load_prop_instances(os.path.join(data_path, f"neg_train_data.txt"))

    if os.path.isfile(embed_path):
        embeddings = load_pretrained_embeddings(embed_path)
    else:
        if vector_type == 'static':
            embeddings = load_pretrained_embeddings(os.path.join(embed_path, 'static_large.txt'))
        elif vector_type == 'glove':
            embeddings = load_pretrained_embeddings(os.path.join(embed_path, 'glove_wikipedia.txt'))
        elif vector_type == 'word2vec':
            embeddings = load_pretrained_embeddings(os.path.join(embed_path, 'word2vec_wikipedia.txt'))
        elif vector_type == 'avg_mention':
            if use_mask:
                fn = 'avg_mv_mask_large_' + max_sent_len + '.txt'
            else:
                fn = 'avg_mv_nomask_large_' + max_sent_len + '.txt'
            embeddings = load_pretrained_embeddings(os.path.join(embed_path, fn))

    logging.info('start to train and test')
    properties = []
    results = []
    cnt = 1
    for prop in pos_train:
        logging.info(str(cnt) + ', ' + prop)
        print(cnt, prop)
        pos_train_data = word_embedding(embeddings, pos_train[prop])
        neg_train_data = word_embedding(embeddings, neg_train[prop])
        pos_test_data = word_embedding(embeddings, pos_test[prop])
        neg_test_data = word_embedding(embeddings, neg_test[prop])
        pos_valid_data = word_embedding(embeddings, pos_valid[prop])
        neg_valid_data = word_embedding(embeddings, neg_valid[prop])

        train_data = np.array(pos_train_data + neg_train_data)
        train_label = np.array([1] * len(pos_train_data) + [-1] * len(neg_train_data))
        test_data = np.array(pos_test_data + neg_test_data)
        test_label = np.array([1] * len(pos_test_data) + [-1] * len(neg_test_data))
        valid_data = np.array(pos_valid_data + neg_valid_data)
        valid_label = np.array([1] * len(pos_valid_data) + [-1] * len(neg_valid_data))
        del pos_train_data
        del neg_train_data
        del pos_test_data
        del neg_test_data
        del pos_valid_data
        del neg_valid_data

        clf, th = train_svc(train_data, valid_data, train_label, valid_label, kernel)
        rr = test_svc(test_data, test_label, clf, th)
        results.append(rr)
        properties.append(prop)
        cnt += 1

    if vector_type == 'static':
        csv_file = os.path.join(result_path, vector_type + "_" + kernel + '_large.csv')
        header = 'svc, ' + vector_type + ', ' + kernel + ',  bert-large-uncased\n'
    elif vector_type == 'avg_mention':
        if use_mask:
            csv_file = os.path.join(result_path, vector_type + '_mask_large_' + max_sent_len + "_" + kernel + '.csv')
            header = 'svc, original, ' + vector_type + ', bert-large-uncased, use mask, max_sent_len = ' + max_sent_len + ', ' + kernel + '\n'
        else:
            csv_file = os.path.join(result_path, vector_type + '_nomask_large_' + max_sent_len + "_" + kernel + '.csv')
            header = 'svc, original, ' + vector_type + ', bert-large-uncased, no mask, max_sent_len = ' + max_sent_len + ", " + kernel + '\n'
    else:
        csv_file = os.path.join(result_path, vector_type + "_" + kernel + '.csv')
        header = 'svc, ' + vector_type + ', ' + kernel + '\n'

    write_csv(csv_file, properties, results)
    write_txt(os.path.join(os.path.abspath(project_path), 'result', dataset + '.txt'), results, header)


if __name__ == '__main__':
    main(sys.argv[1:])
