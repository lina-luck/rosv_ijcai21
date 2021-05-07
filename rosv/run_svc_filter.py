import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
import getopt
from src.utils import load_pretrained_embeddings, init_logging_path, load_prop_instances, write_txt, write_csv
import numpy as np
from src.svm import train_svc, test_svc
import logging
from src.rosv_util import filtered_sample
from sklearn.metrics import average_precision_score


def get_word_embedding(nouns, embeddings_dict, label):
    data = []
    labels = []
    for n in nouns:
        if n in embeddings_dict:
            data.append(embeddings_dict[n])
            labels.append(label)
    return data, labels


def main(argv):
    dataset = 'glasgow'  # mt40k
    use_mask = False
    max_sent_len = '64'
    candidate_k = [115]#, 10, 20, 50, 100, 110, 115]
    kernel = ''
    cluster = 'mean'

    try:
        opts, args = getopt.getopt(argv, "hd:k:m:c:t:", ["data=", "kernel=", "usemask=","tag="])
    except getopt.GetoptError:
        print("python run_svr_filter.py -d <dataset> -k <kernel> -m <use_mask> -t <tag>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("python run_svr_filter.py -d <dataset> -k <kernel> -m <use_mask> -t <tag>")
            sys.exit()
        elif opt in ("-d", "--data"):
            dataset = arg
        elif opt in ("-k", "--kernel"):
            kernel = arg
        elif opt in ("-m", "--usemask"):
            use_mask = bool(arg)
        elif opt in ("-t", "--tag"):
            tag = arg
    print(use_mask)
    data_path = os.path.join(os.path.abspath(project_path), 'data', dataset)
    avg_mv_path = os.path.join(os.path.abspath(project_path), 'data', 'avg_mv_remain')
    if not os.path.exists(data_path) or not os.path.exists(avg_mv_path):
        raise FileNotFoundError

    log_file_path = init_logging_path(os.path.join(os.getcwd(), "log/"), 'run_svc_fb', 'run_svc_fb')
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        )
    logging.info("start logging: " + log_file_path)
    logging.info("dataset is " + dataset)
    logging.info("data path is " + data_path)

    result_path = os.path.join(os.path.abspath(project_path), 'result', dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    logging.info("loading data samples")
    pos_train = load_prop_instances(os.path.join(data_path, f'pos_train_data.txt'))
    pos_valid = load_prop_instances(os.path.join(data_path, f"pos_valid_data.txt"))
    pos_test = load_prop_instances(os.path.join(data_path, f"pos_test_data.txt"))
    neg_train = load_prop_instances(os.path.join(data_path, f"neg_train_data.txt"))
    neg_valid = load_prop_instances(os.path.join(data_path, f"neg_valid_data.txt"))
    neg_test = load_prop_instances(os.path.join(data_path, f"neg_test_data.txt"))
    

    best_map = -1
    best_models = None
    best_ths = None
    best_k = -1
    properties = None
    for k in candidate_k:
        logging.info('k='+str(k))
        models_k = []
        ths_k = []
        map_k = []
        prop_list = []
        if use_mask:
            mv_file = os.path.join(avg_mv_path, tag + '_avg_mv_mask_large_' + dataset + '_' + str(k) + '.txt')
        else:
            mv_file = os.path.join(avg_mv_path, tag + '_avg_mv_nomask_large_' + dataset + '_' + str(k) + '.txt')

        embeddings = load_pretrained_embeddings(mv_file)
        cnt = 1
        for prop in pos_train:
            logging.info(str(cnt)+', '+prop)
            cnt+=1
            train_data, train_label = get_word_embedding(pos_train[prop], embeddings, 1)
            tmp = get_word_embedding(neg_train[prop], embeddings, -1)
            train_data += tmp[0]
            train_label += tmp[1]

            train_data = np.array(train_data)
            train_label = np.array(train_label)

            valid_data, valid_label = get_word_embedding(pos_valid[prop], embeddings, 1)
            tmp = get_word_embedding(neg_valid[prop], embeddings, -1)
            valid_data += tmp[0]
            valid_label += tmp[1]

            valid_data = np.array(valid_data)
            valid_label = np.array(valid_label)

            clf, th = train_svc(train_data, valid_data, train_label, valid_label, kernel)

            pre_score = clf.decision_function(valid_data)
            ap = average_precision_score(valid_label, pre_score)

            models_k.append(clf)
            ths_k.append(th)
            map_k.append(ap)
            prop_list.append(prop)

        map_k = sum(map_k) / len(map_k)

        if map_k > best_map:
            best_k = k
            best_map = map_k
            best_models = models_k
            best_ths = ths_k
            properties = prop_list

    if best_k == -1:
        logging.info('Failed.')
        exit()

    results = []
    i = 0
    if use_mask:
        mv_file = os.path.join(avg_mv_path, tag + '_avg_mv_mask_large_' + dataset + '_' + str(best_k) + '.txt')
    else:
        mv_file = os.path.join(avg_mv_path, tag + '_avg_mv_nomask_large_' + dataset + '_' + str(best_k) + '.txt')

    embeddings = load_pretrained_embeddings(mv_file)

    for prop in properties:
        logging.info(str(i + 1) + ', ' + prop)

        test_data, test_label = get_word_embedding(pos_test[prop], embeddings, 1)
        tmp = get_word_embedding(neg_test[prop], embeddings, -1)
        test_data += tmp[0]
        test_label += tmp[1]

        test_data = np.array(test_data)
        test_label = np.array(test_label)

        rr = test_svc(test_data, test_label, best_models[i], best_ths[i])
        results.append(rr)
        i += 1

    if use_mask:
        csv_file = os.path.join(result_path,
                                tag + '_mask_large_' + max_sent_len + '_' + kernel + '_' + str(best_k) + '.csv')
        header = tag + ', large, mask, best_k = ' + str(best_k) + ', ' + kernel + '\n'
    else:
        csv_file = os.path.join(result_path,
                                tag + '_nomask_large_' + max_sent_len + '_' + kernel + '_' + str(best_k) + '.csv')
        header = tag + ', large, nomask, best_k = ' + str(best_k) + ', ' + kernel + '\n'

    write_csv(csv_file, properties, results)
    write_txt(os.path.join(os.path.abspath(project_path), 'result', dataset + '.txt'), results, header)

    logging.info("Done")


if __name__ == '__main__':
    main(sys.argv[1:])
