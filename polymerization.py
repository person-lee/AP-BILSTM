
import tensorflow as tf
from bilstm import BILSTM
from utils import cal_attention, max_pooling, ortho_weight, uniform_weight, feature2cos_sim, cal_loss_and_acc

class LSTM(object):
    def __init__(self, batch_size, quest_len, answer_len, embeddings, embedding_size, rnn_size, num_rnn_layers, max_grad_norm, l2_reg_lambda=0.0, adjust_weight=False,label_weight=[],is_training=True):
        # define input variable
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.quest_len = quest_len 
        self.answer_len = answer_len 
        self.max_grad_norm = max_grad_norm
        self.l2_reg_lambda = l2_reg_lambda
        self.is_training = is_training

        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        
        self.lr = tf.Variable(0.0,trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

        self.ori_input_quests = tf.placeholder(tf.int32, shape=[None, self.quest_len], name="ori_quest")
        self.cand_input_quests = tf.placeholder(tf.int32, shape=[None, self.answer_len], name="cand_quest")
        self.neg_input_quests = tf.placeholder(tf.int32, shape=[None, self.answer_len], name="neg_quest")
        self.test_input_quests = tf.placeholder(tf.int32, shape=[None, self.quest_len], name="test_quest")
        self.test_input_answer = tf.placeholder(tf.int32, shape=[None, self.answer_len], name="test_cand_quest")

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            ori_quests =tf.nn.embedding_lookup(W, self.ori_input_quests)
            cand_quests =tf.nn.embedding_lookup(W, self.cand_input_quests)
            neg_quests =tf.nn.embedding_lookup(W, self.neg_input_quests)
            test_quest =tf.nn.embedding_lookup(W, self.test_input_quests)
            test_answer =tf.nn.embedding_lookup(W, self.test_input_answer)

        #ori_quests = tf.nn.dropout(ori_quests, self.keep_prob)
        #cand_quests = tf.nn.dropout(cand_quests, self.keep_prob)
        #neg_quests = tf.nn.dropout(neg_quests, self.keep_prob)


        #build LSTM network
        with tf.variable_scope("LSTM_scope", reuse=None):
            ori_q = BILSTM(ori_quests, self.rnn_size)
        with tf.variable_scope("LSTM_scope", reuse=True):
            cand_a = BILSTM(cand_quests, self.rnn_size)
            neg_a = BILSTM(neg_quests, self.rnn_size)
            test_q = BILSTM(test_quest, self.rnn_size)
            test_a = BILSTM(test_answer, self.rnn_size)

        #----------------------------- cal attention -------------------------------
        with tf.variable_scope("attention", reuse=None) as scope:
            U = tf.get_variable("U", [2 * self.rnn_size, 2 * rnn_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            G = tf.batch_matmul(tf.batch_matmul(ori_q, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), cand_a, adj_y=True)
            delta_q = tf.nn.softmax(tf.reduce_max(G, 2))
            delta_a = tf.nn.softmax(tf.reduce_max(G, 1))
            neg_G = tf.batch_matmul(tf.batch_matmul(ori_q, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), neg_a, adj_y=True)
            delta_neg_q = tf.nn.softmax(tf.reduce_max(neg_G, 2))
            delta_neg_a = tf.nn.softmax(tf.reduce_max(neg_G, 1))
        with tf.variable_scope("attention", reuse=True) as scope:
            test_G = tf.batch_matmul(tf.batch_matmul(test_q, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), test_a, adj_y=True)
            delta_test_q = tf.nn.softmax(tf.reduce_max(test_G, 2))
            delta_test_a = tf.nn.softmax(tf.reduce_max(test_G, 1))

        #-------------------------- recalculate lstm output -------------------------
        ori_q_feat = tf.squeeze(tf.batch_matmul(ori_q, tf.reshape(delta_q, [-1, self.quest_len, 1]), adj_x=True))
        cand_q_feat = tf.squeeze(tf.batch_matmul(cand_a, tf.reshape(delta_a, [-1, self.answer_len, 1]), adj_x=True))
        neg_ori_q_feat = tf.squeeze(tf.batch_matmul(ori_q, tf.reshape(delta_neg_q, [-1, self.quest_len, 1]), adj_x=True))
        neg_q_feat = tf.squeeze(tf.batch_matmul(neg_a, tf.reshape(delta_neg_a, [-1, self.answer_len, 1]), adj_x=True))
        test_q_feat = tf.squeeze(tf.batch_matmul(test_q, tf.reshape(delta_test_q, [-1, self.quest_len, 1]), adj_x=True))
        test_a_feat = tf.squeeze(tf.batch_matmul(test_a, tf.reshape(delta_test_a, [-1, self.answer_len, 1]), adj_x=True))

        #-------------------------- recalculate lstm output end ---------------------
        # dropout
        #self.out_ori = tf.nn.dropout(self.out_ori, self.keep_prob)
        #self.out_cand = tf.nn.dropout(self.out_cand, self.keep_prob)
        #self.out_neg = tf.nn.dropout(self.out_neg, self.keep_prob)

        # cal cosine simulation
        self.ori_cand = feature2cos_sim(ori_q_feat, cand_q_feat)
        self.ori_neg = feature2cos_sim(neg_ori_q_feat, neg_q_feat)
        self.test_q_a = feature2cos_sim(test_q_feat, test_a_feat)
        self.loss, self.acc = cal_loss_and_acc(self.ori_cand, self.ori_neg)

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
