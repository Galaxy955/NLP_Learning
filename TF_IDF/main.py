import numpy as np
import itertools
from collections import Counter

docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

# Split the docs.
docs_splited = [d.replace(",", "").split(" ") for d in docs]

# Build the vocab.
vocab = set(itertools.chain(*docs_splited))
v2i = {v: i for i, v in enumerate(vocab)}
i2v = {i: v for i, v in enumerate(vocab)}

def get_idf(method="log"):
    idf_methods = {
        "log": lambda x: 1 + np.log(len(docs) / (x + 1)),
        "prob": lambda x: np.maximum(0, np.log((len(docs) - x) / (x + 1))),
        "len_norm": lambda x: x / (np.sum(np.square(x)) + 1)
    }
    df = np.zeros((len(i2v), 1))
    for i in range(len(i2v)):
        d_count = 0
        for d in docs_splited:
            if i2v[i] in d:
                d_count += 1
        df[i, 0] = d_count
    # print(df)
    idf_fn = idf_methods.get(method)
    if idf_fn is None:
        raise ValueError
    return idf_fn(df)

def get_tf(method="log"):
    tf_methods = {
        "log": lambda x: np.log(1 + x),
        "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
        "boolean": lambda x: np.minimum(x, 1),
    }
    tf = np.zeros((len(vocab), len(docs)), dtype=np.float64)
    for i, d in enumerate(docs_splited):
        counter = Counter(d)
        for v in counter.keys():
            tf[v2i[v], i] = counter[v] / counter.most_common(1)[0][1]
    tf_func = tf_methods.get(method)
    if tf_func is None:
        raise ValueError
    return tf_func(tf)

# Calculate the tf-idf.
idf = get_idf()
tf = get_tf()
tf_idf = tf * idf

# Calculate the cosine similarity.
def cosine_similarity(q, _tf_idf):
    unit_q = q / np.sqrt(np.sum(np.square(q), axis=0, keepdims=True))
    unit_ds = _tf_idf / np.sqrt(np.sum(np.square(_tf_idf), axis=0, keepdims=True))
    similarity = unit_ds.T.dot(unit_q).ravel()
    return similarity

def docs_score(q, len_norm=False):
    q_words = q.replace(",", "").split(" ")

    # add unknown words
    unknown_v = 0
    for v in set(q_words):
        if v not in v2i:
            v2i[v] = len(v2i)
            i2v[len(v2i)-1] = v
            unknown_v += 1
    if unknown_v > 0:
        _idf = np.concatenate((idf, np.zeros((unknown_v, 1), dtype=np.float)), axis=0)
        _tf_idf = np.concatenate((tf_idf, np.zeros((unknown_v, tf_idf.shape[1]), dtype=np.float)), axis=0)
    else:
        _idf, _tf_idf = idf, tf_idf
    counter = Counter(q_words)
    q_tf = np.zeros((len(_idf), 1), dtype=np.float)
    for v in counter.keys():
        q_tf[v2i[v], 0] = counter[v]

    q_vec = q_tf * _idf

    q_scores = cosine_similarity(q_vec, _tf_idf)
    if len_norm:
        len_docs = [len(d) for d in docs_splited]
        q_scores = q_scores / np.array(len_docs)
    return q_scores

q = "I love you"
scores = docs_score(q)
d_ids = scores.argsort()[-3:][::-1]
print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in d_ids]))