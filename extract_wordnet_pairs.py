'''
'''

import codecs
from nltk.corpus import wordnet as wn
import configlogger
from drgriffis.common import log
import dataset

def loadVocabulary(f):
    vocab = set()
    with codecs.open(f, 'r', 'utf-8') as stream:
        for line in stream:
            vocab.add(line.strip())
    return vocab

def enumerateWordNetPairs(vocab, outf, write_lemma=False):
    data = []
    in_vocab = lambda synset: synset.lemmas()[0].name() in vocab
    for pos in ['n', 'v', 'a', 'r']:
        n_pairs = 0
        log.writeln('Processing POS "%s"' % pos)
        log.track(message='  >> Processed {0:,} source synsets ({1:,} pairs)', writeInterval=100)
        for synset in wn.all_synsets(pos):
            if in_vocab(synset):
                for (getter, lbl) in [
                    (synset.hyponyms, dataset.Hyponym),
                    (synset.hypernyms, dataset.Hypernym),
                    (synset.member_holonyms, dataset.Holonym),
                    (synset.substance_holonyms, dataset.Holonym),
                    (synset.part_holonyms, dataset.Holonym),
                    (synset.member_meronyms, dataset.Meronym),
                    (synset.substance_meronyms, dataset.Meronym),
                    (synset.part_meronyms, dataset.Meronym),
                ]:
                    for sink in getter():
                        if in_vocab(sink):
                            if write_lemma:
                                src = synset.lemmas()[0].name()
                                snk = sink.lemmas()[0].name()
                            else:
                                src = synset.name()
                                snk = sink.name()

                            data.append((
                                len(data),
                                src,
                                snk,
                                lbl
                            ))
                            n_pairs += 1
            log.tick(n_pairs)
        log.flushTracker(n_pairs)
        log.writeln('')

    dataset.write(data, outf)

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog VOCABF OUTF')
        parser.add_option('--write-lemma', dest='write_lemma',
                action='store_true', default=False,
                help='write the lemma for the synset instead of the synset ID')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if len(args) != 2:
            parser.print_help()
            exit()
        return args, options
    (vocabf, outf), options = _cli()
    log.start(logfile=options.logfile)

    configlogger.writeConfig(log, [
        ('Vocabulary file to filter to', vocabf),
        ('Output file for relations', outf),
        ('Writing lemmas', options.write_lemma),
    ], 'Filtered WordNet relation generation')

    log.writeln('Reading filter vocab from %s...' % vocabf)
    vocab = loadVocabulary(vocabf)
    log.writeln('Read {0:,} words to filter to.\n'.format(len(vocab)))

    t_sub = log.startTimer('Extracting WordNet pairs....\n')
    enumerateWordNetPairs(vocab, outf, write_lemma=options.write_lemma)
    log.stopTimer(t_sub, message='\nExtraction complete in {0:.2f}s.')

    log.stop()
