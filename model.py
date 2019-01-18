import numpy as np
import tensorflow as tf
from drgriffis.common import log

_SUMMARIZE=12
_DEBUG = False
#_DEBUG = True

class CNNParams:
    
    def __init__(self, 
        embedding_dim=0,
        fully_connected_dim=0,
        num_classes=0,
        num_filters=0,
        #filter_height=0,
        filter_width=0,
        filter_vstride=0,
        filter_hstride=0,
        #pool_height=0,
        pool_width=0,
        #pool_vstride=0,
        pool_hstride=0
    ):
        self.embedding_dim = embedding_dim
        self.fully_connected_dim = fully_connected_dim
        self.num_classes = num_classes

        self.num_filters = num_filters
        #self.filter_height = filter_height
        self.filter_width = filter_width
        self.filter_vstride = filter_vstride
        self.filter_hstride = filter_hstride

        #self.pool_height = pool_height
        self.pool_width = pool_width
        #self.pool_vstride = pool_vstride
        self.pool_hstride = pool_hstride

class CNNModel:
    
    def __init__(self, session, params, checkpoint_path=None):
        self._session = session
        self.p = params
        self._debug = _DEBUG
        self._build()

        self._saver = tf.train.Saver()
        self._checkpoint_path = checkpoint_path

        self._session.run(tf.global_variables_initializer())

    def save(self, fold=None):
        if not fold is None:
            pth = '%s.fold%d' % (self._checkpoint_path, fold)
        else:
            pth = self._checkpoint_path
        self._saver.save(self._session, pth)
    def restore(self, fold=None):
        if not fold is None:
            pth = '%s.fold%d' % (self._checkpoint_path, fold)
        else:
            pth = self._checkpoint_path
        self._saver.restore(self._session, pth)

    def _build(self):
        self._input = tf.placeholder(
            dtype=tf.float32,
            shape=[None, 2, self.p.embedding_dim],
            name='embedding_pair_input'
        )
        self._labels = tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name='labels'
        )
        if self._debug:
            log.writeln(str(self._input))
            log.writeln(str(self._labels))
            _input = tf.Print(self._input, [self._input], summarize=_SUMMARIZE, message='Embedding input')
            labels = tf.Print(self._labels, [self._labels], summarize=_SUMMARIZE, message='Labels')
        else:
            _input = self._input
            labels = self._labels


        conv_filters = tf.Variable(
            tf.truncated_normal(
                [
                    2,  # filter height is always 2
                    self.p.filter_width,
                    1,
                    self.p.num_filters
                ],
                #[2, 2, 1, self.p.num_filters],
                #[2, 2, 1, 1],
                stddev=0.5
            )
        )
        if self._debug:
            log.writeln(str(conv_filters))
            conv_filters = tf.Print(conv_filters, [conv_filters], summarize=_SUMMARIZE, message='Convolutional filters')

        cnn = tf.nn.conv2d(
            input=tf.reshape(
                self._input,
                [-1, 2, self.p.embedding_dim, 1]
            ),
            filter=conv_filters,
            #strides=[1, 1, 1, 1],
            #strides=[1, 1, 2, 1],
            strides=[
                1,
                self.p.filter_vstride,
                self.p.filter_hstride,
                1
            ],
            padding="SAME",
            name='CNN_op'
        )
        if self._debug:
            log.writeln(str(cnn))
            cnn = tf.Print(cnn, [cnn], summarize=_SUMMARIZE, message='CNN output')

        pooled = tf.nn.max_pool(
            value=cnn,
            #ksize=[1, 2, self.p.embedding_dim, 1],
            #strides=[1, 2, self.p.embedding_dim, 1],
            #ksize=[1, 2, 2, 1],
            ksize=[
                1,
                (self.p.filter_vstride % 2) + 1,  # pool height is determined
                                                  # by filter vstride
                self.p.pool_width,
                1
            ],
            #strides=[1, 2, 2, 1],
            strides=[
                1,
                (self.p.filter_vstride % 2) + 1,  # always reduces to 1,
                self.p.pool_hstride,
                1
            ],
            padding='SAME',
            name='max_pooled_CNN'
        )
        if self._debug:
            log.writeln(str(pooled))
            pooled = tf.Print(pooled, [pooled], summarize=_SUMMARIZE, message='Max pooled CNN output')

        #pooled = tf.squeeze(
        #    pooled,
        #    #axis=[1,2]
        #    axis=[1,3]
        #)
        pooled = tf.reshape(
            pooled,
            shape=[
                -1,  # batch_size
                (
                    (self.p.embedding_dim // self.p.pool_hstride)
                    * self.p.num_filters
                )
            ]
        )
        if self._debug:
            log.writeln(str(pooled))
            pooled = tf.Print(pooled, [pooled], summarize=_SUMMARIZE, message='Squeezed pooled')

        pooled = tf.nn.dropout(
            pooled,
            0.6,
            name='pooled_with_dropout'
        )
        if self._debug:
            log.writeln(str(pooled))
            pooled = tf.Print(pooled, [pooled], summarize=_SUMMARIZE, message='Dropout pooled')

        full = tf.contrib.layers.fully_connected(
            pooled,
            self.p.fully_connected_dim,
            activation_fn=tf.nn.relu
        )
        if self._debug:
            log.writeln(str(full))
            full = tf.Print(full, [full], summarize=_SUMMARIZE, message='Fully connected output')

        output_layer = tf.contrib.layers.fully_connected(
            full,
            self.p.num_classes,
            #activation_fn=tf.nn.relu
            activation_fn=None
        )
        if self._debug:
            log.writeln(str(output_layer))
            output_layer = tf.Print(output_layer, [output_layer], summarize=_SUMMARIZE, message='Output layer')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self._labels,
            logits=output_layer,
        )
        self._batch_loss = tf.reduce_sum(
            loss,
        )

        self._scores = tf.nn.softmax(
            output_layer
        )
        self._predictions = tf.argmax(
            self._scores,
            axis=1
        )

        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )
        self._train_step = optimizer.minimize(loss)
    
    def trainStep(self, batch_embeddings, batch_labels):
        feed_dict = {
            self._input: batch_embeddings,
            self._labels: batch_labels
        }
        execs = [self._batch_loss, self._train_step]
        (loss, _) = self._session.run(
            execs,
            feed_dict = feed_dict
        )
        return loss
    
    def getPredictions(self, batch_embeddings, batch_labels):
        feed_dict = {
            self._input: batch_embeddings,
            self._labels: batch_labels
        }
        execs = [self._batch_loss, self._predictions, self._scores]
        results = self._session.run(
            execs,
            feed_dict = feed_dict
        )
        return results
