import os
import sys
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
import getopt
from src.utils import *

def main(argv):
    max_seq_length = 128
    batch_size = 5
    log_dir = os.path.join(os.getcwd(), "log/")

    logfile = 'mv'
    input_dir = '../sents'  # path of sentence files
    output_dir = '../out'
    bert_version = 'roberta-base' #'bert-base-uncased'
    k = -1
    option = ['avg_each', 'mv_last']# ['avg_all', 'avg_k', 'avg_each', 'avg_last', 'mv_last']

    try:
        opts, args = getopt.getopt(argv, "hl:i:b:o:m:k:p:", ["lfile=", "idir=", "batch_size=", "odir=", "model=", "k=", "option="])
    except getopt.GetoptError:
        print('extract_nomask_all.py -l <logfile> -i <input_dir> -b <batch_size> -o <output_dir> -m <bert_model> -k <first_k_layers> -p <option>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('extract_nomask_all.py -l <logfile> -i <input_dir> -b <batch_size> -o <output_dir> -m <bert_model>')
            sys.exit()
        elif opt in ("-l", "--lfile"):
            logfile = arg
        elif opt in ("-i", "--idir"):  # path of sentence files
            input_dir = arg
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-o", "--odir"):  # path to store mention vector files
            output_dir = arg
        elif opt in ("-m", "--model"):
            bert_version = arg
        elif opt in ("-k", "--k"):
            k = int(arg)
        elif opt in ("-p", "--option"):
            option = arg

    if isinstance(option, str):
        option = option.split(',')
    # print(option, type(option))

    num_layer = 12 if 'base' in bert_version else 24
    if option == 'avg_k' and k is None:
        raise ValueError('Please set the value of k when option is avg_k')

    log_file_path = init_logging_path(log_dir, 'extract_nomask_all', logfile)
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        )

    logging.info("start logging: " + log_file_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    usegpu = False
    if torch.cuda.is_available():
        usegpu = True

    logging.info('input dir is ' + input_dir)
    logging.info('output dir is ' + output_dir)
    logging.info('BERT version is ' + bert_version)
    logging.info('batch_size is ' + str(batch_size))

    # store avg vectors from each layer for each noun (24 lines in each file)
    if 'avg_each' in option:
        avg_layer_dir = os.path.join(output_dir, bert_version.split('-')[0] + '_nomask_avg_each_layer')
        if not os.path.exists(avg_layer_dir):
            os.makedirs(avg_layer_dir)

    # store avg of first k layers
    if 'avg_k' in option:
        avg_k_layer_dir = os.path.join(output_dir, bert_version.split('-')[0] + '_nomask_avg_k_layers')
        if not os.path.exists(avg_k_layer_dir):
            os.makedirs(avg_k_layer_dir)

    # store vectors from last layer for each noun, used for filtering
    if 'mv_last' in option:
        mv_dir = os.path.join(output_dir, bert_version.split('-')[0] + '_mv_nomask_last')
        if not os.path.exists(mv_dir):
            os.makedirs(mv_dir)

    cnt = 0
    avg_all_layers = ''
    avg_k_layers = ''
    if 'avg_k' in option and k == -1:
        avg_k_layers = [''] * num_layer
    avg_last_layer = ''
    logging.info("total number of words is " + str(len(os.listdir(input_dir))))
    num_nosent = 0
    for file in os.listdir(input_dir):
        if cnt % 1000 == 0:
            logging.info(str(round(cnt / len(os.listdir(input_dir)) * 100, 2) ) + " % processed")
        logging.info('start extracting mention vectors using nomask model')
        noun = file.split('.')[0]
        if 'avg_each' in option and noun + '.pt' in os.listdir(avg_layer_dir):
            cnt += 1
            continue
        token_ids, input_mask, indices = tokenize_nomask(os.path.join(input_dir, file), max_seq_length, bert_version)
        if len(token_ids) < 1:
            logging.info(noun + " has no sentence")
            num_nosent += 1
            cnt += 1
            continue
        embeddings = extract_mv_nomask(token_ids, input_mask, indices, bert_version=bert_version, batch_size=batch_size, usegpu=usegpu)  # 24 * num_sent * 1024
        # print(embeddings.size())

        # avg of all layers
        if 'avg_all' in option:
            avg_layer = torch.mean(embeddings, dim=0)   # sent_n * dim
            avg_n = torch.mean(avg_layer, dim=0)
            avg_all_layers += ' '.join([noun] + [str(v) for v in avg_n.cpu().numpy()]) + '\n'

        # average of first k layers
        if 'avg_k' in option:
            if k == -1:   # avg for all first k layers iteratively
                for i in range(1, embeddings.shape[0]+1):
                    avg_i = torch.mean(embeddings[:i], dim = 0)
                    avg_n = torch.mean(avg_i, dim=0)
                    avg_k_layers[i-1] += ' '.join([noun] + [str(v) for v in avg_n.cpu().numpy()]) + '\n'
            else:       # avg for first k layers only
                avg_k = torch.mean(embeddings[:k], dim = 0)
                avg_n = torch.mean(avg_k, dim=0)
                avg_k_layers += ' '.join([noun] + [str(v) for v in avg_n.cpu().numpy()]) + '\n'

        # last hidden layer
        if 'avg_last' in option:
            avg_n = torch.mean(embeddings[-1], dim=0)
            avg_last_layer += ' '.join([noun] + [str(v) for v in avg_n.cpu().numpy()]) + '\n'

        if 'mv_last' in option:
            torch.save(embeddings[-1].cpu(), os.path.join(mv_dir, noun + '.pt'))
            # write_csv(os.path.join(mv_dir, noun + '.csv'), embeddings[-1].cpu().numpy())
        # store vectors from each layer for each noun, used for filtering
        elif 'mv_each_layer' in option:
            for l in range(embeddings.shape[0]): # number layer
                l_dir = os.path.join(output_dir, bert_version.split('-')[0] + '_mv_nomask_' + str(l+1))
                if not os.path.exists(l_dir):
                    os.makedirs(l_dir)
                torch.save(embeddings[l].cpu(), os.path.join(l_dir, noun + '.pt'))
                # write_csv(os.path.join(l_dir, noun + '.csv'), embeddings[l].cpu().numpy())

        # avg of each layer individually
        if 'avg_each' in option:
            avg_each_layer = torch.mean(embeddings, dim=1).cpu()  # average among all sentences, shape = (24 * 1024)
            nomask_out_file = os.path.join(avg_layer_dir, noun + '.pt')
            torch.save(avg_each_layer, nomask_out_file)

        cnt += 1

    logging.info(str(num_nosent) + " words have no sentence.")
    logging.info('Done.')

    if 'avg_all' in option and avg_all_layers:
        write_out(os.path.join(output_dir, bert_version.split('-')[0] + '_nomask_avg_all_layers.txt'), avg_all_layers)
    if 'avg_k' in option and avg_k_layers:
        if k == -1:
            for i in range(len(avg_k_layers)):
                write_out(os.path.join(avg_k_layer_dir, bert_version.split('-')[0] + '_nomask_avg_' + str(i+1) + '_layers.txt'), avg_k_layers[i])
        else:
            write_out(os.path.join(avg_k_layer_dir, bert_version.split('-')[0] + '_nomask_avg_' + str(k) + '_layers.txt'), avg_k_layers)
    if 'avg_last' in option and avg_last_layer:
        write_out(os.path.join(output_dir, bert_version.split('-')[0] + '_nomask_avg_last_layer.txt'), avg_last_layer)


if __name__ == '__main__':
    main(sys.argv[1:])
