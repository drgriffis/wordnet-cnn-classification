'''
Methods for working with WordNet triple dataset
'''

import codecs

Hypernym = 0
Hyponym = 1
Meronym = 2
Holonym = 3

Classes = [
    Hypernym,
    Hyponym,
    Meronym,
    Holonym
]

def load(f):
    dataset = []
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            (_id, src, snk, lbl) = [s.strip() for s in line.split('\t')]
            _id, lbl = int(_id), int(lbl)
            dataset.append( (_id, src, snk, lbl) )
    return dataset

def write(data, f):
    with codecs.open(f, 'w', 'utf-8') as stream:
        for (_id, src, snk, lbl) in data:
            stream.write('%s\t%s\t%s\t%d\n' % (
                _id,
                src,
                snk,
                lbl
            ))
