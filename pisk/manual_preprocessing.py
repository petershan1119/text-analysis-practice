import itertools
import os

import gensim
from textblob import TextBlob

fname = "/Users/sangwonhan/practices/python/text_analysis_practice/data/20news-bydate/20news-bydate-test/alt.atheism/53068"

def process_message(message):
    message = gensim.utils.to_unicode(message, 'latin1').strip()
    blocks = message.split(u'\n\n')
    content = u'\n\n'.join(blocks[1:])
    return content


fin = open(fname, 'rb')
content = fin.read()

text = process_message(content)

tokenized_text = list(gensim.utils.tokenize(text, lower=True))

def head(stream, n=10):
    return list(itertools.islice(stream, n))

file_dir = "/Users/sangwonhan/practices/python/text_analysis_practice/data/20news-bydate/20news-bydate-test/alt.atheism"

file_list = os.listdir(file_dir)

text_list = []
for file in file_list[:5]:
    path = os.path.join(file_dir, file)
    with open(path, 'rb') as fin:
        content = fin.read()
    text = process_message(content)
    text_list.append(text)



np_counts = {}
for docno, doc in enumerate(text_list):
    for np in TextBlob(doc).noun_phrases:
        if u' ' not in np:
            continue
        if all(word.isalpha() and len(word) > 2 for word in np.split()):
            np_counts[np] = np_counts.get(np, 0) + 1

    sorted_phrases = sorted(np_counts, key=lambda np: -np_counts[np])