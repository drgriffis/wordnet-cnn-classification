'''
'''

import os
import codecs
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import random
import pyemblib
import configlogger
from drgriffis.common import log
import dataset
import cross_validation
from model import CNNParams, CNNModel

def trainModel(session, model, train, dev, test, embeds,
        batch_size=5, patience=3, early_stopping=0., max_epochs=-1, fold=None,
        preds_file=None, verbose=False, debug=False):

    epoch, best_epoch = 0, -1
    epoch_dev_metrics, current_best = [], 1e9
    training = True
    patience_so_far = 0

    log.writeln('Starting training')
    while training:
        epoch += 1
        batch = 1
        log.writeln('\n== Epoch %d ==\n\n  Training...' % epoch)

        np.random.shuffle(train)

        correct, total = 0., 0
        batch_start = 0
        n_batches = int(np.ceil(len(train) / batch_size))

        if not verbose:
            log.track(message='  >> Processed {0:,}/%s batches' % ('{0:,}'.format(n_batches)), writeInterval=10)

        while (batch_start < len(train)):
            if debug:
                sys.stderr.write('------------------------------\nTRAINING\n')
                sys.stderr.flush()

            next_batch_samples = train[batch_start:batch_start + batch_size]
            next_batch = np.array([
                [
                    embeds[src],
                    embeds[snk]
                ]
                    for (_, src, snk, _) in next_batch_samples
            ])
            batch_labels = np.array([
                lbl for (_, _, _, lbl) in next_batch_samples
            ])

            batch_loss = model.trainStep(
                next_batch,
                batch_labels
            )

            if verbose:
                log.writeln('    Batch {0:,}/{1:,} loss -- {2:.8f}'.format(batch, n_batches, batch_loss))
            else:
                log.tick()

            batch += 1
            batch_start += len(next_batch)

        if not verbose:
            log.flushTracker()

        log.writeln('\n  Training complete.\n  Evaluating on dev...')
        dev_metrics = testModel(
            session,
            model,
            dev,
            embeds,
            batch_size=batch_size,
            preds_file=None,
            training=True,
        )
        log.writeln('    Dev loss -- %f (Best: %f) [Accuracy: %f (%d/%d)]' % (
            dev_metrics.loss, current_best, dev_metrics.accuracy, dev_metrics.correct, dev_metrics.total
        ))

        epoch_dev_metrics.append(dev_metrics)

        # patience/early stopping handling
        if dev_metrics.loss < (current_best - early_stopping):
            patience_so_far = 0
        else:
            patience_so_far += 1
            log.writeln('    >>> Impatience building... (%d/%d) <<<' % (patience_so_far, patience))
            if patience_so_far >= patience:
                log.writeln("    >>> Ran out of patience! <<<")
                log.writeln("           (╯'-')╯︵ ┻━┻ ")
                break

        if dev_metrics.loss < current_best:
            log.writeln('    >>> Improvement! Saving model state. <<<')
            model.save(fold)
            current_best = dev_metrics.loss
            best_epoch = epoch

        if (max_epochs > 0) and epoch >= max_epochs:
            log.writeln("    >>> Hit maximum epoch threshold! <<<")
            log.writeln("                 ¯\(°_o)/¯")
            break

    log.writeln('\nTraining halted.')

    model.restore(fold)

    log.writeln('\nEvaluating best model on train:')
    train_metrics = testModel(
        session,
        model,
        train,
        embeds,
        batch_size=batch_size,
        preds_file=None,
        training=True
    )
    log.writeln('  Accuracy: %f (%d/%d)' % (train_metrics.accuracy, train_metrics.correct, train_metrics.total))

    log.writeln('\nEvaluating best model on test:')
    test_metrics = testModel(
        session,
        model,
        test,
        embeds,
        batch_size=batch_size,
        preds_file=preds_file,
        training=True
    )
    log.writeln('  Accuracy: %f (%d/%d)' % (test_metrics.accuracy, test_metrics.correct, test_metrics.total))

    return best_epoch, epoch_dev_metrics, test_metrics, train_metrics


def testModel(session, model, eval_data, embeds,
        batch_size=5, preds_file=None, training=False,
        debug=False, verbose=False):
    metrics = SimpleNamespace()
    metrics.loss = 0.
    metrics.correct = 0
    metrics.total = 0

    metrics.per_class = {
        dataset.Hypernym: SimpleNamespace(
            tp = 0,
            fp = 0,
            fn = 0
        ),
        dataset.Hyponym: SimpleNamespace(
            tp = 0,
            fp = 0,
            fn = 0
        ),
        dataset.Meronym: SimpleNamespace(
            tp = 0,
            fp = 0,
            fn = 0
        ),
        dataset.Holonym: SimpleNamespace(
            tp = 0,
            fp = 0,
            fn = 0
        )
    }

    batch_start = 0

    while (batch_start < len(eval_data)):
        if debug:
            sys.stderr.write('------------------------------\nTESTING\n')
            sys.stderr.flush()

        next_batch_samples = eval_data[batch_start:batch_start + batch_size]
        next_batch = np.array([
            [
                embeds[src],
                embeds[snk]
            ]
                for (_, src, snk, _) in next_batch_samples
        ])
        batch_labels = np.array([
            lbl for (_, _, _, lbl) in next_batch_samples
        ])

        batch_results = model.getPredictions(
            next_batch,
            batch_labels
        )

        if debug:
            sys.stderr.write('BATCH RESULT: %s\n' % (str(batch_results)))
            sys.stderr.flush()
        (loss, preds, scores) = batch_results
        if str(loss) == 'nan' or str(loss) == 'inf':
            sys.stdout.write('HEY I FOUND ONE\n')
            sys.stdout.flush()
        metrics.loss += np.sum(loss)

        for i in range(len(next_batch)):
            correct_prediction = False

            #found_nan = False
            #for j in range(len(mention.candidates)):
            #    if np.isnan(scores[i][j]):
            #        log.writeln('[WARNING] Found NaN in predicted scores, skipping (Mention ID: %s)' % str(mention.ID))
            #        found_nan = True
            #        break
            #if found_nan:
            #    continue

            # track sample status
            if preds[i] == batch_labels[i]:
                metrics.correct += 1
                metrics.per_class[batch_labels[i]].tp += 1
                correct_prediction = True
            else:
                metrics.per_class[preds[i]].fp += 1
                metrics.per_class[batch_labels[i]].fn += 1
            metrics.total += 1

            # predictions + scores
            if preds_file:
                preds_file.write('Sample %d -- Scores [ %s ] -- Predicted: %d  Gold: %d\n' % (
                    eval_data[batch_start+i][0],
                    ' '.join([str(f) for f in scores[i]]),
                    preds[i],
                    eval_data[batch_start+i][-1]
                ))

        batch_start += len(next_batch)

    if metrics.total > 0:
        metrics.accuracy = float(metrics.correct)/metrics.total
        macro_f1s = []
        for (cls, cls_metrics) in metrics.per_class.items():

            if (cls_metrics.tp + cls_metrics.fp) > 0:
                cls_metrics.precision = (
                    float(cls_metrics.tp) / 
                    (cls_metrics.tp + cls_metrics.fp)
                )
            else:
                cls_metrics.precision = 0

            if (cls_metrics.tp + cls_metrics.fn) > 0:
                cls_metrics.recall = (
                    float(cls_metrics.tp) /
                    (cls_metrics.tp + cls_metrics.fn)
                )
            else:
                cls_metrics.recall = 0

            if (cls_metrics.precision + cls_metrics.recall) > 0:
                cls_metrics.f1 = (
                    (2 * cls_metrics.precision * cls_metrics.recall) /
                    (cls_metrics.precision + cls_metrics.recall)
                )
            else:
                cls_metrics.f1 = 0

            macro_f1s.append(cls_metrics.f1)
        metrics.macro_f1 = np.mean(macro_f1s)
    else:
        metrics.accuracy = 0
        metrics.macro_f1 = 0

    return metrics

def preprocess(ds, embeddings, options):
    preprocessed = SimpleNamespace()

    preprocessed.samples = ds
    preprocessed.embeddings = embeddings

    preprocessed.samples_by_id = {
        s[0] : s
            for s in preprocessed.samples
    }
    preprocessed.labels_by_id = {
        s[0] : s[-1]
            for s in preprocessed.samples
    }

    preprocessed.splits = cross_validation.crossValidationSplits(
        preprocessed.labels_by_id,
        n_folds=options.n_folds,
        dev_size=options.dev_size,
        persistent_path=options.cross_validation_file,
        random_seed=options.random_seed,
        log=log
    )

    return preprocessed

def runCrossValidationExperiment(preprocessed, options, preds_file):
    params = CNNParams(
        embedding_dim = preprocessed.embeddings.size,
        fully_connected_dim = options.fully_connected_dim,
        num_classes = len(dataset.Classes),
        num_filters = options.num_filters,
        filter_width = options.filter_width,
        filter_vstride = options.filter_vstride,
        filter_hstride = options.filter_hstride,
        pool_width = options.pool_width,
        pool_hstride = options.pool_hstride
    )

    train_metrics, test_metrics, best_dev_metrics, best_epochs = [], [], [], []
    if options.dev_results_file:
        options.dev_results_file = open(options.dev_results_file, 'w')

    for i in range(len(preprocessed.splits)):
        log.writeln(('\n\n{0}\n  Starting fold %d/%d\n{0}\n'.format('#'*80)) % (i+1, len(preprocessed.splits)))

        np.random.seed(options.random_seed + i)
        tf.set_random_seed(options.random_seed + i)
        tf.reset_default_graph()

        with tf.Session() as session:
            cnn = CNNModel(
                session,
                params,
                checkpoint_path=options.checkpoint_path
            )

            (train_ids, dev_ids, test_ids) = preprocessed.splits[i]
            train, dev, test = [], [], []
            for _id in train_ids:
                if _id in preprocessed.samples_by_id:
                    train.append(preprocessed.samples_by_id[_id])
            for _id in dev_ids:
                if _id in preprocessed.samples_by_id:
                    dev.append(preprocessed.samples_by_id[_id])
            for _id in test_ids:
                if _id in preprocessed.samples_by_id:
                    test.append(preprocessed.samples_by_id[_id])

            best_epoch, fold_dev_metrics, fold_test_metrics, fold_train_metrics = trainModel(
                session,
                cnn,
                train,
                dev,
                test,
                preprocessed.embeddings,
                batch_size=options.batch_size,
                patience=options.patience,
                early_stopping=options.early_stopping,
                max_epochs=options.max_epochs,
                fold=i,
                preds_file=preds_file
            )

            if options.dev_results_file:
                options.dev_results_file.write('%d\tTotalLoss\t%s\n' % (i, ','.join([str(m.loss) for m in fold_dev_metrics])))
                options.dev_results_file.write('%s\tAccuracies\t%s\n' % (i, ','.join([str(m.accuracy) for m in fold_dev_metrics])))

            train_metrics.append(fold_train_metrics)
            test_metrics.append(fold_test_metrics)
            best_dev_metrics.append(fold_dev_metrics[best_epoch - 1])  # epoch starts at 1, fold_dev_metrics at 0
            best_epochs.append(best_epoch)

            del(cnn)

    for (lbl, metric_set) in [('train', train_metrics), ('dev', best_dev_metrics), ('test', test_metrics)]:
        log.writeln('\n\n-- Cross-validation report (%s) --\n' % lbl)
        log.writeln('Accuracy/# epochs to convergence per fold:')
        for i in range(len(metric_set)):
            m = metric_set[i]
            log.writeln('  Fold %d -- Accuracy: %f (%d/%d) [%d epochs]' % (i+1, m.accuracy, m.correct, m.total, best_epochs[i]))

        log.writeln('\nOverall cross-validation accuracy (%s): %f' % (lbl, np.mean([m.accuracy for m in metric_set])))

    if options.dev_results_file:
        options.dev_results_file.close()

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog DATASET')
        parser.add_option('--embeddings', dest='embedding_f',
                help='(REQUIRED) input embedding file')
        parser.add_option('--batch-size', dest='batch_size',
                help='batch size for execution (default: %default)',
                type='int', default=5)
        parser.add_option('--n-fold', dest='n_folds',
                type='int', default=10,
                help='number of folds for cross validation (default: %default)')
        parser.add_option('--dev-size', dest='dev_size',
                type='float', default=0.1,
                help='portion of cross-validation training data to hold back for development'
                     ' (default %default; must be >0 and <1)')
        parser.add_option('--cross-validation-splits', dest='cross_validation_file',
                help='path to save cross-validation splits to (generates multiple files; optional)')
        parser.add_option('--patience', dest='patience',
                type='int', default=3,
                help='number of epochs to wait for improvement before halting;'
                     ' if set to 0, halts as soon as dev performance stops improving'
                     ' (default: %default)')
        parser.add_option('--early-stopping', dest='early_stopping',
                type='float', default=1e-5,
                help='threshold for determining when dev performance has stopped improving'
                     ' (default: %default)')
        parser.add_option('--max-epochs', dest='max_epochs',
                type='int', default=1000,
                help='maximum number of epochs to train (default %default; -1 for no limit)')
        parser.add_option('--checkpoint-file', dest='checkpoint_path',
                default='/tmp/model.ckpt',
                help='path to checkpoint file for saving model during training (default: %default)')
        parser.add_option('--random-seed', dest='random_seed',
                type='int', default=-1,
                help='random seed for reproducibility (defaults to epoch time)')
        parser.add_option('--predictions', dest='predictions_file',
                help='file to write predicted scores to')
        parser.add_option('--dev-results', dest='dev_results_file',
                help='file to write dev results to')

        group = optparse.OptionGroup(parser, 'CNN settings')
        group.add_option('--fully-connected-dim', dest='fully_connected_dim',
                type='int', default=800,
                help='dimension for fully connected layer')
        group.add_option('--num-filters', dest='num_filters',
                type='int', default=100,
                help='number of convolutional filters')
        group.add_option('--filter-width', dest='filter_width',
                type='int', default=2,
                help='width of convolutional filters (default %default)')
        group.add_option('--filter-vstride', dest='filter_vstride',
                type='int', default=1,
                help='vertical stride of convolutional filters (must be <= 2; default %deffault)')
        group.add_option('--filter-hstride', dest='filter_hstride',
                type='int', default=1,
                help='horizontal stride of convolutional filters (default %default)')
        group.add_option('--pool-width', dest='pool_width',
                type='int', default=2,
                help='max pooling width (default %default)')
        group.add_option('--pool-hstride', dest='pool_hstride',
                type='int', default=2,
                help='max pooling horizontal stride (default %default)')

        parser.add_option('-l', '--logfile', dest='logfile',
                help='name of file to write log contents to (empty for stdout)',
                default=None)
        (options, args) = parser.parse_args()

        # guarantee no overlap of model checkpoint files in concurrent experiments
        # note this does NOT use options.random_seed intentionally; does not affect
        # reproducibility, just means that model files won't overwrite one another
        random_checkpoint_identifier = random.randint(0, 100000)
        if options.checkpoint_path:
            options.checkpoint_path = '%s_%d' % (options.checkpoint_path, random_checkpoint_identifier)

        if not options.predictions_file:
            options.predictions_file = '%s.predictions' % os.path.splitext(options.logfile)[0]
        if not options.dev_results_file:
            options.dev_results_file = '%s.dev_results' % os.path.splitext(options.logfile)[0]
        
        now_stamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
        if options.logfile:
            options.logfile = '%s.%s' % (options.logfile, now_stamp)
        if options.predictions_file:
            options.predictions_file = '%s.%s' % (options.predictions_file, now_stamp)
        if options.dev_results_file:
            options.dev_results_file = '%s.%s' % (options.dev_results_file, now_stamp)
        if options.checkpoint_path:
            options.checkpoint_path = '%s.%s' % (options.checkpoint_path, now_stamp)

        if options.random_seed < 0:
            options.random_seed = int(time.time())

        if (
            len(args) != 1
            or (not options.embedding_f)
        ):
            parser.print_help()
            exit()
        return args, options

    (dataset_f,), options = _cli()
    log.start(logfile=options.logfile)
    configlogger.writeConfig(log, [
        ('Dataset file', dataset_f),
        ('Word embeddings', options.embedding_f),
        ('Batch size', options.batch_size),
        ('CNN settings', [
            ('# convolutional filters', options.num_filters),
            ('Filter width', options.filter_width),
            ('Filter v-stride', options.filter_vstride),
            ('Filter h-stride', options.filter_hstride),
            ('Pooling width', options.pool_width),
            ('Pooling h-stride', options.pool_hstride),
            ('Fully connected dimension', options.fully_connected_dim),
        ]),
        ('Training settings', [
            ('Patience', options.patience),
            ('Early stopping criterion', options.early_stopping),
            ('Max training epochs', options.max_epochs),
            ('Checkpoint file', options.checkpoint_path),
            ('Cross validation splits file', options.cross_validation_file),
            ('Number of folds', options.n_folds),
            ('Fraction of training used for dev', options.dev_size),
            ('Writing predictions to', options.predictions_file),
            ('Writing dev results to', options.dev_results_file),
            ('Random seed', options.random_seed),
        ]),
    ], 'WordNet classification experiment')

    t_sub = log.startTimer('Reading word embeddings from %s...' % options.embedding_f)
    embeddings = pyemblib.read(options.embedding_f)
    log.stopTimer(t_sub, message='Read {0:,} embeddings ({1}s).\n'.format(
        len(embeddings), '{0:.2f}'
    ))

    log.writeln('Reading dataset from %s...' % dataset_f)
    ds = dataset.load(dataset_f)
    log.writeln('Read {0:,} samples.\n'.format(len(ds)))

    preprocessed = preprocess(ds, embeddings, options)

    if options.predictions_file:
        preds_stream = codecs.open(options.predictions_file, 'w', 'utf-8')

    runCrossValidationExperiment(
        preprocessed,
        options,
        preds_stream
    )

    if options.predictions_file:
        preds_stream.close()

    log.stop()
