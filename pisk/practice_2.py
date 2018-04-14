import logging
import tarfile

import gensim
import itertools

import re

from gensim.parsing.preprocessing import STOPWORDS

import nltk
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from textblob import TextBlob

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

with tarfile.open('/Users/sangwonhan/practices/python/text_analysis_practice/data/20news-bydate.tar.gz', 'r:gz') as tf:
    file_infos = [file_info for file_info in tf if file_info.isfile()]

    message = tf.extractfile(file_infos[0]).read()
    print(message)

def process_message(message):
    message = gensim.utils.to_unicode(message, 'latin1').strip()
    blocks = message.split(u'\n\n')
    content = u'\n\n'.join(blocks[1:])
    return content

print(process_message(message))


def iter_20newsgroups(fname, log_every=None):
    """
    Yield plain text of each 20 newsgroups message, as a unicode string.

    The messages are read from raw tar.gz file `fname` on disk (e.g. `./data/20news-bydate.tar.gz`)

    """
    extracted = 0
    with tarfile.open(fname, 'r:gz') as tf:
        for file_number, file_info in enumerate(tf):
            if file_info.isfile():
                if log_every and extracted % log_every == 0:
                    logging.info("extracting 20newsgroups file #%i: %s" % (extracted, file_info.name))
                content = tf.extractfile(file_info).read()
                yield process_message(content)
                extracted += 1


message_stream = iter_20newsgroups('/Users/sangwonhan/practices/python/text_analysis_practice/data/20news-bydate.tar.gz', log_every=2)
print(list(itertools.islice(message_stream, 3)))


class Corpus20News(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for text in iter_20newsgroups(self.fname):
            yield list(gensim.utils.tokenize(text, lower=True))


tokenized_corpus = Corpus20News('/Users/sangwonhan/practices/python/text_analysis_practice/data/20news-bydate.tar.gz')


class Corpus20News_Lemmatize(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for message in iter_20newsgroups(self.fname):
            yield self.tokenize(message)

    def tokenize(self, text):
        return gensim.utils.lemmatize(text, stopwords=STOPWORDS)

lemmatized_corpus = Corpus20News_Lemmatize('/Users/sangwonhan/practices/python/text_analysis_practice/data/20news-bydate.tar.gz')
print(list(itertools.islice(lemmatized_corpus, 2)))


def best_ngrams(words, top_n=1000, min_freq=100):
    """
    Extract `top_n` most salient collocations (bigrams and trigrams),
    from a stream of words. Ignore collocations with frequency
    lower than `min_freq`.

    This fnc uses NLTK for the collocation detection itself -- not very scalable!

    Return the detected ngrams as compiled regular expressions, for their faster
    detection later on.

    """
    tcf = TrigramCollocationFinder.from_words(words)
    tcf.apply_freq_filter(min_freq)
    trigrams = [' '.join(w) for w in tcf.nbest(TrigramAssocMeasures.chi_sq, top_n)]
    logging.info("%i trigrams found: %s..." % (len(trigrams), trigrams[:20]))

    bcf = tcf.bigram_finder()
    bcf.apply_freq_filter(min_freq)
    bigrams = [' '.join(w) for w in bcf.nbest(BigramAssocMeasures.pmi, top_n)]
    logging.info("%i bigrams found: %s..." % (len(bigrams), bigrams[:20]))

    pat_gram2 = re.compile('(%s)' % '|'.join(bigrams), re.UNICODE)
    pat_gram3 = re.compile('(%s)' % '|'.join(trigrams), re.UNICODE)

    return pat_gram2, pat_gram3


class Corpus20News_Collocations(object):
    def __init__(self, fname):
        self.fname = fname
        logging.info("collecting ngrams from %s" % self.fname)
        # generator of documents; one element = list of words
        documents = (self.split_words(text) for text in iter_20newsgroups(self.fname, log_every=1000))
        # generator: concatenate (chain) all words into a single sequence, lazily
        words = itertools.chain.from_iterable(documents)
        self.bigrams, self.trigrams = best_ngrams(words)

    def split_words(self, text, stopwords=STOPWORDS):
        """
        Break text into a list of single words. Ignore any token that falls into
        the `stopwords` set.

        """
        return [word
                for word in gensim.utils.tokenize(text, lower=True)
                if word not in STOPWORDS and len(word) > 3]

    def tokenize(self, message):
        """
        Break text (string) into a list of Unicode tokens.

        The resulting tokens can be longer phrases (collocations) too,
        e.g. `new_york`, `real_estate` etc.

        """
        text = u' '.join(self.split_words(message))
        text = re.sub(self.trigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        text = re.sub(self.bigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        return text.split()

    def __iter__(self):
        for message in iter_20newsgroups(self.fname):
            yield self.tokenize(message)


collocations_corpus = Corpus20News_Collocations('/Users/sangwonhan/practices/python/text_analysis_practice/data/20news-bydate.tar.gz')

def head(stream, n=10):
    return list(itertools.islice(stream, n))

def best_phrases(document_stream, top_n=1000000, prune_at=1000000):
    np_counts = {}
    for docno, doc in enumerate(document_stream):
        if docno % 1000 == 0:
            sorted_phrases = sorted(np_counts.items(), key=lambda item: -item[1])
            np_counts = dict(sorted_phrases[:prune_at])
            logging.info("at document #%i, considering %i phrases: %s..." % (docno, len(np_counts), head(sorted_phrases)))

        for np in TextBlob(doc).noun_phrases:
            if u' ' not in np:
                continue

            if all(word.isalpha() and len(word) > 2 for word in np.split()):
                np_counts[np] = np_counts.get(np, 0) + 1

    sorted_phrases = sorted(np_counts, key=lambda np: -np_counts[np])
    return set(head(sorted_phrases, top_n))

fname = '/Users/sangwonhan/practices/python/text_analysis_practice/data/20news-bydate.tar.gz'


class Corpus20News_NE(object):
    def __init__(self, fname):
        self.fname = fname
        logging.info("collecting entities from %s" % self.fname)
        doc_stream = itertools.islice(iter_20newsgroups(self.fname), 1000)
        # doc_stream = iter_20newsgroups(self.fname)
        self.entities = best_phrases(doc_stream)
        logging.info("selected %i entities: %s..." %
                     (len(self.entities), list(self.entities)[:10]))

    def __iter__(self):
        for message in iter_20newsgroups(self.fname):
            yield self.tokenize(message)

    def tokenize(self, message, stopwords=STOPWORDS):
        """
        Break text (string) into a list of Unicode tokens.

        The resulting tokens can be longer phrases (named entities) too,
        e.g. `new_york`, `real_estate` etc.

        """
        result = []
        for np in TextBlob(message).noun_phrases:
            if u' ' in np and np not in self.entities:
                # only consider multi-word phrases we detected in the constructor
                continue
            token = u'_'.join(part for part in gensim.utils.tokenize(np) if len(part) > 2)
            if len(token) < 2 or token in stopwords:
                # ignore very short phrases and stop words
                continue
            result.append(token)
        return result

ne_corpus = Corpus20News_NE(fname)
print(head(ne_corpus, 5))