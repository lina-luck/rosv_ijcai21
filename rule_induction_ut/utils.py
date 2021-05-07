"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import numpy as np
import torch
import dgl
import os
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from graph import read_dictionary
from collections import defaultdict


def init_logging_path(log_path, task_name, file_name):
    dir_log  = os.path.join(log_path,f"{task_name}/{file_name}/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    return dir_log


def select_threshold_rules(y_true, y_prob, path, fold, idx, train_label):
    print(type(y_true), y_true.shape, np.where(y_true.numpy().sum(axis=0) > 0))
    interested_template_ids = np.where(y_true.numpy().sum(axis=0) > 0)[1]
    thresholds = np.unique(y_prob)
    best_f1 = -1
    best_th = -1
    for th in thresholds:
        y_pred = (y_prob> th) * np.ones_like(y_prob)
        true_rules, _ = find_rules_ut(path, fold, idx, train_label, y_true, interested_template_ids)
        pred_rules, _ = find_rules_ut(path, fold, idx, train_label, y_pred, interested_template_ids, pred=True)
        print(len(set(true_rules)), len(set(pred_rules)))
        f1 = metrics(set(true_rules), set(pred_rules))[2]

        if best_f1 < f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1


#######################################################################
#
# Utility function for finding predicted rules
#
#######################################################################
def find_rules_ut_all(path, train_label, probality, topk):
    # print(pred)
    node_dict_file = path + 'train/s1' + '/all_nodes.dict'  # for all data
    template_dict_file = path + 'train/s1' + '/all_unary_templates.dict'  # for all data
    result = ''

    node_dict = read_dictionary(node_dict_file)  # {id: name}
    template_dict = read_dictionary(template_dict_file)  # {id: template name}

    probality[np.where(train_label == 1)] = 0  # ignore the true rules

    topk_idx = probality.reshape(-1).argsort()[::-1][0:topk]

    for idx in topk_idx:
        n_id = idx // probality.shape[1]
        t_id = idx % probality.shape[1]
        if probality[n_id, t_id] <= 0:
            break
        n_name = node_dict[n_id]
        tmp_name = template_dict[int(t_id)]
        rule = tmp_name.replace('TempateExpression', 'ER')
        rule = rule.replace('?', n_name)
        result += rule + '\n'
    return result

## find rules for unary templates
# dataset: wine, sumo, ...
# template_ids: unary template ids
def find_rules_ut(path, fold, node_ids, train_label, template_ids, interested_template_ids, pred=False, easytask=False):
    # print(pred)
    node_dict_file = path + 'train/s' + str(fold + 1) + '/nodes.dict'  # for all data
    template_dict_file = path + 'train/s' + str(fold + 1) + '/unary_templates.dict'  # for all data
    rules = []
    result = ''

    selected_templates = np.where(train_label.sum(axis=0) > 2)[1]

    node_dict = read_dictionary(node_dict_file)  # {id: name}
    template_dict = read_dictionary(template_dict_file)  # {id: template name}

    for idx in range(len(node_ids)):
        n_id = node_ids[idx]
        n_name = node_dict[n_id]
        # if pred:
        tmp_ids = np.where(template_ids[idx] > 0)[0]
        # else:
        #     tmp_ids = template_ids[idx]
        if not tmp_ids.size:
            continue
        for id in tmp_ids:
            if easytask:
                t_id = selected_templates[id]
            else:
                t_id = id
            if train_label[n_id, t_id] == 1:
                continue
            if pred and t_id not in interested_template_ids:
                continue
            tmp_name = template_dict[int(t_id)]
            rule = tmp_name.replace('TempateExpression', 'ER')
            rule = rule.replace('?', n_name)
            rules.append(rule)
            result += rule + '\n'

    return rules, result

## find rules for binary templates
# dataset: wine, economy,...
# triples: test triples with id, i.e. (n_id, bt_id, n_id)
# pred_templates: predicated bt_ids, note: there may be more than one bt id for each node pair
def find_rules_bt(dataset, type, triples, pred_templates, pred=False):
    node_dict_file = 'dataset/' + type + '/' + dataset + '/nodes.dict'
    template_dict_file = 'dataset/' + type + '/' + dataset + '/edges.dict'
    result = ''

    node_dict = read_dictionary(node_dict_file)  # {id: name}
    template_dict = read_dictionary(template_dict_file)  # {id: template name}
    s = triples[:,0]
    r = triples[:,1]
    t = triples[:,2]
    for idx in range(len(s)):
        s_id = s[idx]
        t_id = t[idx]
        s_name = node_dict[int(s_id)]
        t_name = node_dict[int(t_id)]
        if pred:
            tmp_ids = np.where(pred_templates[idx] > 0)[0]  # for pred label
            if not tmp_ids.size:
                continue
        else:
            tmp_ids = r[idx]  # for true label
        for tid in tmp_ids:
            tmp_name = template_dict[tid]
            result += s_name + '\t' + tmp_name[:-1] + '\t' + t_name + '\n'

    return result

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):   # i can be regarded as edge_id
        adj_list[triplet[0]].append([i, triplet[2]])
        #adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list]) # out degree
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """ Edge neighborhood sampling to reduce training graph size
    """
    #print(sample_size)
    edges = np.zeros(sample_size, dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])  # sample count for each node
    picked = np.array([False for _ in range(n_triplets)])   # num_triple * 1
    seen = np.array([False for _ in degrees])  # num_node * 1
    for i in range(0, sample_size):
        weights = sample_counts * (~seen)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0
        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        #print(chosen_vertex)
        while len(adj_list[chosen_vertex]) == 0:
            chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                             p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]   # can be regarded as edge_id

        while picked[edge_number]:  # if edge has been choosed before, then choose again
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True
    return edges

def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    edges = sample_edge_neighborhood(adj_list, degrees, len(triplets),
                                     sample_size)

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    print("# sampled nodes: {}".format(len(uniq_v)))
    print("# sampled edges: {}".format(len(src) * 2))
    g, rel, norm, edge_norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels, edge_norm

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph.
        some edges are binary, but others single
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)

    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)

    #src, rel, dst = triplets
    #g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))

    # normalize by dst degree, compute degrees according to edge_type
    _, inverse_index, count = np.unique((dst, rel), axis=1, return_inverse=True,
                                        return_counts=True)
    degrees = count[inverse_index]  # c_{i,r} for each relation type
    edge_norm = np.ones(len(dst), dtype=np.float32) / degrees.astype(np.float32)
    return g, rel, norm, edge_norm

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate) # node_id
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

#######################################################################
#
# Utility function for evaluations
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def perturb_and_get_rank(embedding, w, a, r, b, num_entity, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (num_entity + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        #print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(num_entity, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        #print('embedding[batch_a] * w[batch_r]', emb_ar.shape)
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        #print('emb_ar.transpose(0, 1).unsqueeze(2)', emb_ar.shape)
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        #print('emb_c', emb_c.shape)
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        #print(out_prod.shape)
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        print('score', score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)


# TODO (lingfan): implement filtered metrics
# return MRR (raw), and Hits @ (1, 3, 10)
def evaluate(test_graph, model, test_triplets, num_entity, hits=[], eval_bz=100):
    with torch.no_grad():
        embedding, w = model.evaluate(test_graph)
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]

        #print(w[0])
        #print(w.detach().numpy().shape)
        #print(embedding[o])
        # perturb subject
        ranks_s = perturb_and_get_rank(embedding, w, o, r, s, num_entity, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_rank(embedding, w, s, r, o, num_entity, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        #print("MRR (raw): {:.6f}".format(mrr.item()))

        hit_n = []
        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            hit_n.append(avg_count.item())
            #print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item(), hit_n

# TODO: implement my own metrics
# return (micro-)precision, recall, F1, mAP
def compute_score(test_graph, model, test_triplets):
    with torch.no_grad():
        embedding, w = model.evaluate(test_graph)
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]

        emb_s = embedding[s]  # num_node * num_fea
        emb_t = embedding[t]
        #print(w[0])
        #print(emb_t)

        emb_s = emb_s.transpose(0, 1).unsqueeze(2)  # num_fea * num_node * 1
        w = w.transpose(0, 1).unsqueeze(1)  # num_fea * 1 * num_rel
        mult_sr = torch.bmm(emb_s, w)  # emb_s * w, size = num_fea * num_node * num_rel

        mult_sr = mult_sr.transpose(1, 2)  # num_fea * num_rel * num_node
        mult_sr = mult_sr.transpose(0, 2)  # num_node * num_rel * num_fea
        emb_t = emb_t.unsqueeze(2)  # num_node * num_fea * 1

        products = torch.bmm(mult_sr, emb_t)  # num_node * num_rel * 1
        products = products.squeeze(2)  # num_node * num_rel
        score = torch.sigmoid(products)  # num_node * num_rel

        
        y_true = label_binarize(r, classes=np.arange(w.size()[2]))  # num_node * num_relss
        #print(y_true)
        #print(score)
    return score, y_true 

# select best threshold
def select_threshold(y_true, y_prob, num_class):
    best_thred = []
    for i in range(num_class):
        thresholds = sorted(set(y_prob[:,i]))
        f1 = []
        for th in thresholds:
            y_pred = (y_prob[:,i] > th) * np.ones_like(y_prob[:,i])
            f1.append(f1_score(y_true[:,i], y_pred, average='micro'))
        best_thred.append(thresholds[int(np.argmax(np.array(f1)))])
    return best_thred

def _select_threshold(y_true, y_prob):
    # print(y_true.shape)
    thresholds = np.array(range(1, 1000))/1000.0  #sorted(set(y_prob.reshape(-1)))
    f1 = []
    for th in thresholds:
        y_pred = (y_prob> th) * np.ones_like(y_prob)
        #true_rules, _ = find_rules_ut(path, fold, idx, train_label, y_true, interested_template_ids)
        # pred_rules, _ = find_rules_ut(path, fold, idx, train_label, y_pred, interested_template_ids)
        # f1.append(metrics(set(true_rules), set(pred_rules))[2])
        f1.append(f1_score(y_true, y_pred, average='micro'))
    best_thred = thresholds[int(np.argmax(np.array(f1)))]
    print(max(f1))
    return best_thred, max(f1)

def metrics(test_rules, pred_rules):
    acc_num = len(test_rules.intersection(pred_rules))

    # for r in range(len(set(test_rules))):
    #     if test_rules[r] in pred_rules:
    #         # print(test_rules[r])
    #         acc_num += 1

    precision = acc_num / (len(set(pred_rules)) + 1e-10)
    recall = acc_num / (len(set(test_rules)) + 1e-10)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-10)
    print(acc_num, len(set(test_rules)), len(set(pred_rules)))
    #map = average_precision_score(test_rule_labels, pred_rule_labels)
    return precision, recall, f1


# find edges with same type
# return tuple like (etype, list of index)
def find_same_etype(etype):
    tally = defaultdict(list)
    for i, item in enumerate(etype):
        tally[int(item)].append(i)

    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)
