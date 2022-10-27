import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

corpus = [
    # Numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # Alphabets.
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]

# Pre-process the corpus.
corpus = [x.split(" ") for x in corpus]
cbow = Word2Vec(corpus, vector_size=100, window=3, min_count=0, sg=0, epochs=300)
skip_gram = Word2Vec(corpus, vector_size=100, window=3, min_count=0, sg=1, epochs=300)
indexes = list(set([n for m in corpus for n in m]))
indexes.sort()
corpus_vector_cbow = {i: cbow.wv[i] for i in indexes}
corpus_vector_skipgram = {i: skip_gram.wv[i] for i in indexes}

# Visualize the result of Word2Vec.
all_vectors_cbow = np.array([corpus_vector_cbow[x] for x in indexes])
all_vectors_skipgram = np.array([corpus_vector_skipgram[x] for x in indexes])
pca = PCA(n_components=2)
pca.fit(all_vectors_cbow)
corpus_vector_cbow = {i: pca.transform(corpus_vector_cbow[i].reshape(1, 100)) for i in indexes}
pca.fit(all_vectors_skipgram)
corpus_vector_skipgram = {i: pca.transform(corpus_vector_skipgram[i].reshape(1, 100)) for i in indexes}
fig = plt.figure(figsize=(8, 8), dpi=200)
fig.add_subplot(2, 1, 1)
plt.xlim(-0.5, 0.7)
plt.ylim(-0.3, 0.5)
plt.title("CBOW")
for i in range(10):
    plt.text(corpus_vector_cbow[indexes[i]][0][0], corpus_vector_cbow[indexes[i]][0][1], s=indexes[i], c="b")
for i in range(10, len(indexes)):
    plt.text(corpus_vector_cbow[indexes[i]][0][0], corpus_vector_cbow[indexes[i]][0][1], s=indexes[i], c="r")
fig.add_subplot(2, 1, 2)
plt.xlim(-0.6, 1.2)
plt.ylim(-0.2, 0.2)
plt.title("Skip-gram")
for i in range(10):
    plt.text(corpus_vector_skipgram[indexes[i]][0][0], corpus_vector_skipgram[indexes[i]][0][1], s=indexes[i], c="b")
for i in range(10, len(indexes)):
    plt.text(corpus_vector_skipgram[indexes[i]][0][0], corpus_vector_skipgram[indexes[i]][0][1], s=indexes[i], c="r")
# plt.show()
plt.savefig("result.jpg")