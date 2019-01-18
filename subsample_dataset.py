'''
Load in all generated WordNet triples and subsample equal-sized
chunks from each relation type.
'''

import dataset
import numpy as np
import time
import configlogger
from drgriffis.common import log

def collateByClass(ds):
    collated = {}
    for (_id, src, snk, lbl) in ds:
        if not lbl in collated:
            collated[lbl] = []
        collated[lbl].append( (_id, src, snk, lbl) )
    return collated

def subsample(collated, size):
    sampled = []
    for (lbl, subset) in collated.items():
        subset = subset.copy()
        np.random.shuffle(subset)
        sampled.extend(subset[:size])
    return sampled

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog --input DS_IN --output DS_OUT')
        parser.add_option('-i', '--input', dest='inputf',
                help='(REQUIRED) full dataset file')
        parser.add_option('-o', '--output', dest='outputf',
                help='(REQUIRED) output subsampled dataset file')
        parser.add_option('--size', dest='size',
                type='int', default=5000,
                help='number of samples per class (default %default)')
        parser.add_option('--random-seed', dest='random_seed',
                type='int', default=-1,
                help='random seed for subsampling')
        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()

        if options.random_seed < 0:
            options.random_seed = int(time.time())

        if (not options.inputf) or (not options.outputf):
            parser.print_help()
            exit()
        return options
    options = _cli()
    log.start(logfile=options.logfile)
    configlogger.writeConfig(log, [
        ('Input file', options.inputf),
        ('Output file', options.outputf),
        ('# samples per class', options.size),
        ('Random seed', options.random_seed),
    ], 'WordNet dataset subsampling')

    log.writeln('Reading dataset from %s...' % options.inputf)
    ds = dataset.load(options.inputf)
    log.writeln('Read {0:,} samples.\n'.format(len(ds)))

    log.writeln('Collating by class...')
    collated = collateByClass(ds)
    classes = list(collated.keys())
    classes.sort()
    for c in classes:
        log.writeln('  {0} --> {1:,}'.format(c, len(collated[c])))
        if len(collated[c]) < options.size:
            log.writeln('[WARNING] subsample size is too large for class "{0}"'.format(c))

    log.writeln('\nSubsampling...')
    np.random.seed(options.random_seed)
    subsampled = subsample(collated, options.size)
    log.writeln('Subsampled {0:,} instances.\n'.format(len(subsampled)))

    log.writeln('Writing subsampled dataset to %s...' % options.outputf)
    dataset.write(subsampled, options.outputf)
    log.writeln('Done.\n')
    
    log.stop()
