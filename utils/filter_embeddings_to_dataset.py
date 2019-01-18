'''
Given a pre-generated WordNet relations dataset, filter a set of
word embeddings to only those keys used in the dataset.

Purely for efficiency reasons in loading in the word embeddings.
'''

import codecs
import pyemblib
import configlogger
from drgriffis.common import log
import dataset

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-i', '--input', dest='inputf',
                help='(REQUIRED) unfiltered word embeddings')
        parser.add_option('-o', '--output', dest='outputf',
                help='(REQUIRED) file to write filtered word embeddings to')
        parser.add_option('-d', '--dataset', dest='datasetf',
                help='(REQUIRED) pre-generated dataset for filtering')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()
        if (not options.inputf) or (not options.outputf) or (not options.datasetf):
            parser.print_help()
            exit()
        return options
    options = _cli()
    log.start(logfile=options.logfile)
    configlogger.writeConfig(log, [
        ('Input embeddings file', options.inputf),
        ('Output embeddings file', options.outputf),
        ('Dataset file', options.datasetf),
    ], 'Embedding filtering for WordNet classification experiments')

    t_sub = log.startTimer('Reading input embeddings from %s...' % options.inputf)
    embeddings = pyemblib.read(options.inputf)
    log.stopTimer(t_sub, message='Read {0:,} embeddings in {1}s.\n'.format(
        len(embeddings), '{0:.2f}'
    ))

    log.writeln('Reading vocabulary from dataset in %s...' % options.datasetf)
    ds = dataset.load(options.datasetf)
    vocab = set()
    for (_, src, snk, _) in ds:
        vocab.add(src)
        vocab.add(snk)
    log.writeln('Found {0:,} unique words in {1:,} samples.\n'.format(
        len(vocab), len(ds)
    ))

    log.writeln('Filtering embeddings...')
    filtered = pyemblib.Embeddings()
    for (k,v) in embeddings.items():
        if k in vocab:
            filtered[k] = v
    log.writeln('Reduced to {0:,} embeddings.\n'.format(len(filtered)))

    log.writeln('Writing filtered embeddings to %s...' % options.outputf)
    pyemblib.write(filtered, options.outputf, verbose=True)
    log.writeln('Done.\n')

    log.stop()
