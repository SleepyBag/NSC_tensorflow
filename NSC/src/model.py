import tensorflow as tf


class NSC(object):

    def __init__(self, max_sen_len, max_doc_len, cls_cnt, emb_file,
                 emb_dim, sen_hidden_size, doc_hidden_size, usr_cnt, prd_cnt,
                 usr_hidden_size, prd_hidden_size):
        self.max_sen_len = max_sen_len
        self.max_doc_len = max_doc_len
        self.cls_cnt = cls_cnt
        self.emb_file = emb_file
        self.emb_dim = emb_dim
        self.sen_hidden_size = sen_hidden_size
        self.doc_hidden_size = doc_hidden_size
        self.usr_cnt = usr_cnt
        self.prd_cnt = prd_cnt
        self.usr_hidden_size = usr_hidden_size
        self.prd_hidden_size = prd_hidden_size

        with tf.name_scope('inputs'):
            self.usrid = tf.placeholder(tf.int32, [None], name="usrid")
            self.prdid = tf.placeholder(tf.int32, [None], name="prdid")
            self.input_x = tf.placeholder(tf.int32, [None, self.max_doc_len, self.max_sen_len], name="input_x")
            self.input_y = tf.placeholder(tf.int32, [None, self.cls_cnt], name="input_y")
            self.sen_len = tf.placeholder(tf.int32, [None, self.max_doc_len], name="sen_len")
            self.doc_len = tf.placeholder(tf.int32, [None], name="doc_len")

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': tf.Variable(tf.random_uniform([self.doc_hidden_size, self.cls_cnt], -.01, .01)),

                'sen_wh': tf.Variable(tf.random_uniform([self.sen_hidden_size, sen_hidden_size], -.01, .01)),
                'sen_wu': tf.Variable(tf.random_uniform([self.usr_hidden_size, sen_hidden_size], -.01, .01)),
                'sen_wp': tf.Variable(tf.random_uniform([self.prd_hidden_size, sen_hidden_size], -.01, .01)),
                'sen_v': tf.Variable(tf.random_uniform([self.sen_hidden_size, 1], -.01, .01)),

                'doc_wh': tf.Variable(tf.random_uniform([self.doc_hidden_size, doc_hidden_size], -.01, .01)),
                'doc_wu': tf.Variable(tf.random_uniform([self.usr_hidden_size, doc_hidden_size], -.01, .01)),
                'doc_wp': tf.Variable(tf.random_uniform([self.prd_hidden_size, doc_hidden_size], -.01, .01)),
                'doc_v': tf.Variable(tf.random_uniform([self.doc_hidden_size, 1], -.01, .01))
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': tf.Variable(tf.random_uniform([self.cls_cnt], -.01, .01)),

                'sen_attention_b': tf.Variable(tf.random_uniform([self.sen_hidden_size], -.01, .01)),
                'doc_attention_b': tf.Variable(tf.random_uniform([self.doc_hidden_size], -.01, .01))
            }

        with tf.name_scope('emb'):
            self.wrd_emb = tf.constant(self.emb_file, name='wrd_emb', dtype=tf.float32)
            self.x = tf.nn.embedding_lookup(self.wrd_emb, self.input_x)
            self.usr_emb = tf.Variable(tf.random_uniform([self.usr_cnt, self.usr_hidden_size], -.01, .01), dtype=tf.float32)
            self.prd_emb = tf.Variable(tf.random_uniform([self.prd_cnt, self.prd_hidden_size], -.01, .01), dtype=tf.float32)
            self.usr = tf.nn.embedding_lookup(self.usr_emb, self.usrid)
            self.prd = tf.nn.embedding_lookup(self.prd_emb, self.prdid)

    def attention(self, v, wh, h, wu, u, wp, p, b):
        # h = tf.matmul(tf.reshape(h, [-1, hidden_size]), wh)
        # u = tf.matmul(tf.reshape(u, [-1, self.usr_hidden_size]), wu)
        # p = tf.matmul(tf.reshape(p, [-1, self.prd_hidden_size]), wp)
        h_shape = h.shape
        batch_size = h_shape[0]
        max_doc_len = h_shape[1]
        hidden_size = h_shape[2]
        with tf.name_scope('attention'):
            h = tf.reshape(h, [-1, hidden_size])
            h = tf.matmul(h, wh)
            e = tf.reshape(h + b, [-1, max_doc_len, hidden_size])
            # e = tf.reshape(e, [-1, max_doc_len, hidden_size])
            u = tf.matmul(u, wu)
            p = tf.matmul(p, wp)
            e = e + u[:, None, :] + p[:, None, :]
            e = tf.tanh(e)
            e = tf.reshape(e, [-1, hidden_size])
            e = tf.reshape(tf.matmul(e, v), [-1, max_doc_len])[:, None, :]
            # e = tf.reshape(e, [batch_size, 1, max_doc_len])
            e = tf.nn.softmax(e)
        return e

    def lstm(self, inputs, sequence_length, hidden_size, scope):
        outputs, state = tf.nn.dynamic_rnn(
            cell=tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.),
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            scope=scope
        )
        return outputs, state

    def nsc(self):
        inputs = tf.reshape(self.x, [-1, self.max_sen_len, self.emb_dim])
        sen_len = tf.reshape(self.sen_len, [-1])

        with tf.name_scope('sentence'):
            outputs, state = self.lstm(inputs, sen_len, self.sen_hidden_size, 'sentence')
            alpha = self.attention(self.weights['sen_v'], self.weights['sen_wh'],
                                   outputs, self.weights['sen_wu'], self.usr,
                                   self.weights['sen_wp'], self.prd, self.biases['sen_attention_b'])
            outputs = tf.matmul(alpha, outputs)
            outputs = tf.reshape(outputs, [-1, self.max_doc_len, self.sen_hidden_size])

        with tf.name_scope('doc'):
            outputs, state = self.lstm(outputs, self.doc_len, self.doc_hidden_size, 'doc')
            beta = self.attention(self.weights['doc_v'], self.weights['doc_wh'],
                                  outputs, self.weights['doc_wu'], self.usr,
                                  self.weights['doc_wp'], self.prd, self.biases['doc_attention_b'])
            outputs = tf.matmul(beta, outputs)
            d = tf.reshape(outputs, [-1, self.sen_hidden_size])

        with tf.name_scope('result'):
            d_hat = tf.tanh(tf.matmul(d, self.weights['softmax']) + self.biases['softmax'])
            p_hat = tf.nn.softmax(d_hat)
        return p_hat

    def build(self):
        self.p_hat = self.nsc()
        self.prediction = tf.argmax(self.p_hat, 1, name='predictions')

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.p_hat, labels=self.input_y)

        with tf.name_scope("metrics"):
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.mse = tf.reduce_sum(tf.square(self.prediction - tf.argmax(self.input_y, 1)), name="mse")
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.int32), name="correct_num")
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

        return loss, self.mse, self.correct_num, self.accuracy
