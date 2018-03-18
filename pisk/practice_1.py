import gensim
import nltk
import numpy
import textblob

import itertools

gensim.utils.lemmatize("The quick brown fox jumps over the lazy dog!")

nltk.download('brown')

textblob.TextBlob("The quick brown fox jumps over the lazy dog!").noun_phrases


def odd_numbers():
    result = 1
    while True:
        yield result
        result += 2

odd_numbers_generator = odd_numbers()


for odd_number in odd_numbers_generator:
    print(odd_number)
    if odd_number > 10:
        break

class OddNumbers(object):
    def __iter__(self):
        result = 1
        while True:
            yield result
            result += 2

x = numpy.random.rand(10, 5)

print(x[2,1])
print(x[2])
print(x[:, 1])
print(x[:4, :2])


infinite_stream = OddNumbers()

print(list(itertools.islice(infinite_stream, 10)))

concat_stream = itertools.chain('abcde', infinite_stream)
print(list(itertools.islice(concat_stream, 10)))
