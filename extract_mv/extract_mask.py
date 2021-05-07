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
    bert_version = 'roberta-base'
    option = ['mv_last', 'avg_last']

    try:
        opts, args = getopt.getopt(argv, "hl:i:b:o:m:p:", ["lfile=", "idir=", "batch_size=", "odir=", "model=", "option="])
    except getopt.GetoptError:
        print('extract_mask.py -l <logfile> -i <input_dir> -b <batch_size> -o <output_dir> -m <bert_model> -p <option>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('extract_mask.py -l <logfile> -i <input_dir> -b <batch_size> -o <output_dir> -m <bert_model> -p <option>')
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
        elif opt in ("-p", "--option"):
            option = arg

    if isinstance(option, str):
        option = option.split(',')

    log_file_path = init_logging_path(log_dir, 'extract_mask', logfile)
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        )

    logging.info("start logging: " + log_file_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if 'mv_last' in option:
        mv_dir = os.path.join(output_dir, bert_version.split('-')[0] + '_mv_mask_last')
        if not os.path.exists(mv_dir):
            os.makedirs(mv_dir)

    usegpu = False
    if torch.cuda.is_available():
        usegpu = True

    logging.info('input dir is ' + input_dir)
    logging.info('output dir is ' + output_dir)
    logging.info('BERT version is ' + bert_version)
    logging.info('batch_size is ' + str(batch_size))

    cnt = 0
    avg_last_layer = ''
    logging.info("total number of words is " + str(len(os.listdir(input_dir))))
    # print(len(os.listdir(input_dir)))
    num_nosent = 0
    for file in os.listdir(input_dir):
        if cnt % 1000 == 0:
            logging.info(str(round(cnt / len(os.listdir(input_dir)) * 100, 2) ) + " % processed")
        logging.info('start extracting mention vectors using mask model')
        noun = file.split('.')[0]
        if 'mv_last' in option and noun + '.pt' in os.listdir(mv_dir):
            cnt += 1
            continue
        token_ids, input_mask, indices = tokenize_mask(os.path.join(input_dir, file), max_seq_length, bert_version)
        if len(token_ids) < 1:
            logging.info(noun + " has no sentence")
            num_nosent += 1
            cnt += 1
            continue
        embeddings = extract_mv_mask(token_ids, input_mask, indices, bert_version=bert_version, batch_size=batch_size, usegpu=usegpu)  # 24 * num_sent * 1024


        # store vectors from each layer for each noun, used for filtering
        if 'mv_each_layer' in option:
            for l in range(embeddings.shape[0]):  # number layer
                l_dir = os.path.join(output_dir, bert_version.split('-')[0] + '_mv_mask_' + str(l + 1))
                if not os.path.exists(l_dir):
                    os.makedirs(l_dir)
                # write_csv(os.path.join(l_dir, noun + '.csv'), embeddings[l].cpu().numpy())
                torch.save(embeddings[l].cpu(), os.path.join(l_dir, noun + '.pt'))

        if 'mv_last' in option:
            torch.save(embeddings[-1].cpu(), os.path.join(mv_dir, noun + '.pt'), _use_new_zipfile_serialization=False)
            # write_csv(os.path.join(mv_dir, noun + '.csv'), embeddings[-1].cpu().numpy())

        if 'avg_last' in option:
            # last hidden layer
            avg_n = torch.mean(embeddings[-1], dim=0)
            avg_last_layer += ' '.join([noun] + [str(v) for v in avg_n.cpu().numpy()]) + '\n'
        cnt += 1
    logging.info(str(num_nosent) + " words have no sentence.")
    logging.info('Done.')

    if 'avg_last' in option and avg_last_layer:
        write_out(os.path.join(output_dir, bert_version.split('-')[0] + '_mask_avg_last_layer.txt'), avg_last_layer)


if __name__ == '__main__':
    main(sys.argv[1:])
