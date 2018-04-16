import nltk
from nltk import DefaultTagger, RegexpTagger, UnigramTagger, BigramTagger, TrigramTagger
from nltk.corpus import treebank
from pattern.text import parsetree

sentence = 'The brown fox is quick and he is jumping over the lazy dog'

tokens = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokens, tagset='universal')

data = treebank.tagged_sents()
train_data = data[:3500]
test_data = data[3500:]

tokens = nltk.word_tokenize(sentence)

dt = DefaultTagger('NN')

dt.evaluate(test_data)

patterns = [
        (r'.*ing$', 'VBG'),               # gerunds
        (r'.*ed$', 'VBD'),                # simple past
        (r'.*es$', 'VBZ'),                # 3rd singular present
        (r'.*ould$', 'MD'),               # modals
        (r'.*\'s$', 'NN$'),               # possessive nouns
        (r'.*s$', 'NNS'),                 # plural nouns
        (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
        (r'.*', 'NN')                     # nouns (default) ...
]

rt = RegexpTagger(patterns)

rt.evaluate(test_data)

ut = UnigramTagger(train_data)
bt = BigramTagger(train_data)
tt = TrigramTagger(train_data)

ut.evaluate(test_data)

def combined_tagger(train_data, taggers, backoff=None):
    for tagger in taggers:
        backoff = tagger(train_data, backoff=backoff)
    return backoff

ct = combined_tagger(train_data=train_data, taggers=[UnigramTagger, BigramTagger, TrigramTagger], backoff=rt)

tree = parsetree(sentence)

for sentence_tree in tree:
    print(sentence_tree.chunks)

