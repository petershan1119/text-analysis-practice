import logging
import re
import os

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