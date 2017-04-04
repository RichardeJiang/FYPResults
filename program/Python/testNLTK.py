import nltk

sentence = "a good girl with a good tatoo"

tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)

grammar = r"""IP:{<DT>?<JJ|NN|NNS>*<NN|NNS><IN><DT>?<JJ|NN|NNS>*}
NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}"""
grammar1 = r"""NP:{<DT>?<JJ|NN|NNS>*<NN|NNS>}
IP:{<DT>?<JJ|NN|NNS>*<NN|NNS><IN><DT>?<JJ|NN|NNS>*}"""
cp = nltk.RegexpParser(grammar1)
result = cp.parse(tagged)
print len(result)
print result[0].label()