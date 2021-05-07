from gensim.models import KeyedVectors
import numpy as np
import re

def sep_by_uppercase(str):
    pattern = "[A-Z]"
    new_str = re.sub(pattern, lambda x: " " + x.group(0), str)
    if str[0].islower():  # first character is lowercase
        return new_str
    else:
        return new_str[1:]

def word_embedding(embeding_file, nodes_dict):
    dim = 300
    embeddings = KeyedVectors.load_word2vec_format(embeding_file, binary=True)
    node_embeding = np.zeros((len(nodes_dict),dim))  # default value is 0
    for nod in nodes_dict:
        words = nod.strip().split('#') # [1][:-1]
        # print(words)
        if len(words) == 1:
            words = words[0].split('/')[-1][:-1]
            # print(words)
        else:
            words = words[1][:-1]
            # print(words)
        words = sep_by_uppercase(words)
        if words in embeddings:
            node_embeding[nodes_dict[nod]] = embeddings[words]
        else:
            if len(words) > 1:  # multi-words
                words = words.split(' ')
                vectors = np.zeros((len(words),dim))
                for w in words:
                    if w in embeddings:
                        vectors[words.index(w)] = embeddings[w]
                node_embeding[nodes_dict[nod]] = np.mean(np.array(vectors), axis=0)
    return node_embeding