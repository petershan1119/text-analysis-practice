import logging
import re
import os

import numpy as np

import itertools

import gensim
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess, smart_open

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

def head(stream, n=10):
    return list(itertools.islice(stream, n))


def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_wiki(dump_file):
    ignore_namespaces = "Wikipedia Category File Portal Template MediaWiki User Help Book Draft".split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue
        yield title, tokens

fname = '/Users/sangwonhan/practices/python/text_analysis_practice/data/simplewiki-20180301-pages-articles.xml.bz2'

stream = iter_wiki(fname)

for title, tokens in itertools.islice(iter_wiki(fname), 8):
    print(title, tokens[:10])

id2word = {0: u'word', 2: u'profit', 300: u'another_word'}

doc_stream = (tokens for _, tokens in iter_wiki(fname))

id2word_wiki = gensim.corpora.Dictionary(doc_stream)

id2word_wiki.filter_extremes(no_below=20, no_above=0.1)

doc = "A blood cell, also called a hematocyte, is a cell produced by hematopoiesis and normally found in blood."

bow = id2word_wiki.doc2bow(tokenize(doc))


class WikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs


wiki_corpus = WikiCorpus(fname, id2word_wiki)

vector = next(iter(wiki_corpus))
print(vector)
most_index, most_count = max(vector, key=lambda set: set[1])

for id, _ in vector:
    print(id2word_wiki[id])

print(id2word_wiki[most_index], most_count)


gensim.corpora.MmCorpus.serialize('/Users/sangwonhan/practices/python/text_analysis_practice/data/wiki_bow.mm', wiki_corpus)

mm_corpus = gensim.corpora.MmCorpus('/Users/sangwonhan/practices/python/text_analysis_practice/data/wiki_bow.mm')


clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, 4000)

lda_model = gensim.models.LdaModel(clipped_corpus, num_topics=10, id2word=id2word_wiki, passes=4)


tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word_wiki)
lsi_model = gensim.models.LsiModel(tfidf_model[mm_corpus], id2word=id2word_wiki, num_topics=200)

text = "A blood cell, also called a hematocyte, is a cell produced by hematopoiesis and normally found in blood."

bow_vector = id2word_wiki.doc2bow(tokenize(text))

print([(id2word_wiki[id], count) for id, count in bow_vector])

lda_vector = lda_model[bow_vector]
print(lda_vector)
print(lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))

lsi_vector = lsi_model[tfidf_model[bow_vector]]
print(lsi_vector)
print(lsi_model.print_topic(max(lsi_vector, key=lambda item: abs(item[1]))[0]))

lda_model.save('/Users/sangwonhan/practices/python/text_analysis_practice/data/lda_wiki.model')
lsi_model.save('/Users/sangwonhan/practices/python/text_analysis_practice/data/lsi_wiki.model')
tfidf_model.save('/Users/sangwonhan/practices/python/text_analysis_practice/data/tfidf_wiki.model')
id2word_wiki.save('/Users/sangwonhan/practices/python/text_analysis_practice/data/wiki.dictionary')

top_words = [[word for word, coef in lda_model.show_topic(topicno, topn=50)] for topicno in range(lda_model.num_topics)]

all_words = set(itertools.chain.from_iterable(top_words))

replace_index = np.random.randint(0, 10, lda_model.num_topics)

replacements = []
for topicno, words in enumerate(top_words):
    other_words = all_words.difference(words)
    replacement = np.random.choice(list(other_words))
    replacements.append((words[replace_index[topicno]], replacement))
    words[replace_index[topicno]] = replacement
    print("%i: %s" % (topicno, ' '.join(words[:10])))