from dgl import DGLGraph
import argparse
import torch.nn.functional as F
from layers import RGCNBasisLayer as RGCNLayer
from model import BaseRGCN
from utils import _select_threshold, find_rules_ut, metrics, find_same_etype, init_logging_path
import random
from data_preprocessing import train_test_idx
from itertools import combinations
import warnings
import logging
from pytorchtools import *

warnings.filterwarnings('ignore')

random.seed(123)

class UnaryClassify(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.in_feat)
        if self.use_cuda:
            features = features.to('cuda')
        return features

    def build_input_layer(self):
        return RGCNLayer(self.in_feat, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True, node_features=self.features)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, dropout=self.dropout,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=None)


# similar loss
def similar_loss(g, node_idx, logits, direction='in'):
    all_loss = 0
    for i in node_idx:
        if direction == 'in':
            u, _, eid = g.in_edges(i, 'all')
        else:
            _, u, eid = g.out_edges(i, 'all')
        etype = g.edata['type'][eid]
        loss = 0
        for idx in find_same_etype(etype):
            sm_nodes = u[idx[1]]
            cnt = 0
            l = 0
            for n1, n2 in combinations(sm_nodes, 2):
                cnt += 1
                l += torch.norm(logits[n1] - logits[n2], 1)
            loss = l / cnt
        all_loss += loss
    return all_loss / len(node_idx)


# def my_loss(g, node_idx, logits, true_label):
#     sim_loss_in = similar_loss(g, no)
#
#     return


def main(args):
    log_file_path = init_logging_path('log', 'ut', args.dataset)
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    if not os.path.exists('result/'):
        os.mkdir('result/')

    if not os.path.exists('rules/'):
        os.mkdir('rules/')

    if args.dataset == 'sumo':
        folds = [3]
    else:
        folds = [10]
    randint = random.randint(1, 10000)
    for fold in folds:
        result = []
        sigma = args.sigma
        beta = args.beta
        output = str(args) + '\n'
        true_rules_out = ''
        pred_rules_out = ''
        for i in range(fold):
            logging.info("fold " + str(i+1))
            path = 'dataset/' + args.dataset + '/' + str(fold) + '_fold' + '/'
            num_node, edge_list, edge_src, edge_dst, edge_type, edge_norm, num_rel, train_idx, test_idx, train_label, test_label, node_features = train_test_idx(path, i, args.ftype, args.k)

            tmp_label = train_label
            train_idx = list(train_idx)
            test_template_id = np.where(test_label.sum(axis=0) > 0)[1]
            #print(test_label.shape, test_template_id)

            if test_label.shape[0] == 0:
                continue
            edge_type = torch.from_numpy(edge_type).long()
            edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
            node_features = torch.from_numpy(np.array(node_features)).float()

            # check cuda
            use_cuda = args.gpu and torch.cuda.is_available()
            if use_cuda:
                #torch.cuda.set_device('gpu')
                edge_type = edge_type.to('cuda')
                edge_norm = edge_norm.to('cuda')
                node_features = node_features.to('cuda')

            # create multi-graph
            g = DGLGraph(multigraph=True)
            g.add_nodes(num_node)
            g.add_edges(edge_src, edge_dst)
            if use_cuda:
                g = g.to('cuda')
            g.edata.update({'type': edge_type, 'norm': edge_norm})

            # node_norm for apply_func
            in_deg = g.in_degrees(range(g.number_of_nodes())).float().cpu().numpy()
            norm = 1.0 / in_deg
            norm[np.isinf(norm)] = 0
            node_norm = torch.from_numpy(norm)
            if use_cuda:
                node_norm = node_norm.to('cuda')
            g.ndata.update({'norm': node_norm})

            in_feat = node_features.shape[1]
            num_classes = train_label.shape[1]

            train_label = torch.from_numpy(np.array(train_label)).float()
            test_label = torch.from_numpy(test_label.toarray()).float()
            # create model
            model = UnaryClassify(in_feat,
                              args.n_hidden,
                              num_classes,
                              num_rel,
                              num_bases=args.n_bases,
                              num_hidden_layers=args.n_layers - 2,
                              dropout=args.dropout,
                              use_cuda=use_cuda,
                              features=node_features)
            if use_cuda:
                model.to('cuda')
                train_label.to('cuda')
                test_label.to('cuda')


            # optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')  #
            model_idx = len(os.listdir('trained_model')) + 1
            model_name = 'trained_model/checkpoint_' + str(model_idx+randint) + '.pt'
            early_stopping = EarlyStopping(patience=30, verbose=False, path=model_name)

            if args.validation:
                val_idx = sorted(random.sample(list(train_idx), num_node // 5))  # 10% for validation
                train_idx = sorted(list(set(train_idx) - set(val_idx)))
            else:
                val_idx = train_idx
            val_labels = train_label[val_idx]
            train_label = train_label[train_idx]
            logging.info("start training...")
            model.train()
            for epoch in range(args.n_epochs):
                optimizer.zero_grad()
                logits = model.forward(g)
                # similar loss
                sim_loss_in = similar_loss(g, train_idx, logits, 'in')
                sim_loss_out = similar_loss(g, train_idx, logits, 'out')
                classify_loss = criterion(logits[train_idx].cpu(), train_label.cpu())  # for multi-label classification
                loss = classify_loss + sigma * sim_loss_in + beta * sim_loss_out
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()

                model.eval()
                sim_loss_in = similar_loss(g, val_idx, logits, 'in')
                sim_loss_out = similar_loss(g, val_idx, logits, 'out')
                classify_loss = criterion(logits[val_idx].cpu(), val_labels.cpu())  # for multi-label classification
                val_loss = classify_loss + sigma * sim_loss_in + beta * sim_loss_out
                #print(epoch+1, loss.item(), val_loss.item())
                logging.info(str(epoch+1) + ', ' + str(round(loss.item(), 5)) + ', ' + str(round(val_loss.item(), 5)))
                model.train()
                early_stopping(val_loss.item(), model)
                if early_stopping.early_stop:
                    print("Stop. Model trained")
                    break

            model.load_state_dict(torch.load(model_name))
            model.eval()
            logits = model.forward(g).cpu()
            logits = F.sigmoid(logits)
            best_thred, _ = _select_threshold(val_labels.cpu(), logits[val_idx].detach().cpu().numpy())

            labels_test_pre = np.zeros(test_label.shape)
            labels_test_pre[np.where(logits[test_idx].detach().cpu().numpy() > best_thred)] = 1

            test_rules, str_terule = find_rules_ut(path, i, test_idx, tmp_label, test_label.cpu(), test_template_id)
            pred_rules, str_prerule = find_rules_ut(path, i, test_idx, tmp_label, labels_test_pre, test_template_id, pred=True)

            true_rules_out += 'fold ' + str(i) + '\n'
            true_rules_out += str_terule
            pred_rules_out += 'fold ' + str(i) + '\n'
            pred_rules_out += str_prerule

            test_precision, test_recall, test_f1 = metrics(set(test_rules), set(pred_rules))

            result.append([test_precision, test_recall, test_f1])
            print("Test Precision: {:.4f} | Test Recall: {:.4f} | Test F1: {:.4f}".format(test_precision, test_recall, test_f1))
            output += "Test Precision: {:.4f} | Test Recall: {:.4f} | Test F1: {:.4f} \n".format(test_precision, test_recall, test_f1)

            if args.dataset == 'sumo':
                break
        mean_p, mean_r, mean_f1 = np.mean(np.array(result), axis=0)

        print("Mean values over " + str(fold) + " fold: Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}".format(mean_p, mean_r, mean_f1))
        output += "Mean values over " + str(fold) + " fold: Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}\n\n\n\n".format(mean_p, mean_r, mean_f1)

        file_name = args.ftype + '_' + args.dataset + '_' + str(fold) + '.txt'

        f = open('result/' + file_name, 'a+', encoding='utf-8')
        f.write(output)
        f.close()

        f = open('rules/true_' + file_name, 'a+', encoding='utf-8')
        f.write(true_rules_out)
        f.close()

        f = open('rules/pred_' + file_name, 'a+', encoding='utf-8')
        f.write(pred_rules_out)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden units")
    parser.add_argument("--gpu", action='store_true', default=False,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.001,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='wine',
            help="dataset to use")
    parser.add_argument("-f", "--ftype", type=str, default='embedding',
            help="feature type to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="similar loss in coef")
    parser.add_argument("--beta", type=float, default=0,
                        help="similar loss out coef")
    parser.add_argument("--k", type=int, default=5,
                        help="number of neighbors")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = 0
    main(args)
