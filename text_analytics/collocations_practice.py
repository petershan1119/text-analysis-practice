import itertools
from operator import itemgetter

import nltk
from nltk.collocations import *
from nltk.corpus import gutenberg

from utils.normalization import normalize_corpus, parse_document

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))

finder.nbest(bigram_measures.pmi, 10)


alice = gutenberg.sents(fileids='carroll-alice.txt')
alice = [' '.join(ts) for ts in alice]
norm_alice = normalize_corpus(alice, lemmatize=False)

def flatten_corpus(corpus):
    return ' '.join([document.strip() for document in corpus])

def compute_ngrams(sequence, n):
    return zip(*[sequence[index:] for index in range(n)])


def get_top_ngrams(corpus, ngram_val=1, limit=5):
    corpus = flatten_corpus(corpus)
    tokens = nltk.word_tokenize(corpus)

    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(), key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq) for text, freq in sorted_ngrams]
    return sorted_ngrams

get_top_ngrams(corpus=norm_alice, ngram_val=2, limit=10)


finder = BigramCollocationFinder.from_documents([item.split() for item in norm_alice])

bigram_measures = nltk.collocations.BigramAssocMeasures()


toy_text = """
Elephants are large mammals of the family Elephantidae 
and the order Proboscidea. Two species are traditionally recognised, 
the African elephant and the Asian elephant. Elephants are scattered 
throughout sub-Saharan Africa, South Asia, and Southeast Asia. Male 
African elephants are the largest extant terrestrial animals. All 
elephants have a long trunk used for many purposes, 
particularly breathing, lifting water and grasping objects. Their 
incisors grow into tusks, which can serve as weapons and as tools 
for moving objects and digging. Elephants' large ear flaps help 
to control their body temperature. Their pillar-like legs can 
carry their great weight. African elephants have larger ears 
and concave backs while Asian elephants have smaller ears 
and convex or level backs.  
"""

def get_chunks(sentences, grammer = r'NP: {<DT>? <JJ>* <NN.*>+}'):
    all_chunks = []
    chunker = nltk.chunk.regexp.RegexpParser(grammer)

    for sentence in sentences:
        tagged_sents = nltk.pos_tag_sents([nltk.word_tokenize(sentence)])
        chunks = [chunker.parse(tagged_sent) for tagged_sent in tagged_sents]
        wtc_sents = [nltk.chunk.tree2conlltags(chunk) for chunk in chunks]
        flatten_chunks = list(itertools.chain.from_iterable(wtc_sent for wtc_sent in wtc_sents))
sentences = parse_document(toy_text)
