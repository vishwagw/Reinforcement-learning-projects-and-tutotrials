#in this file, we are going to crrate the brain of our A.I agent.
#all the decision process are being made here
#context - 1 --> check README.md

#libraires and suites
#Tenssorflow
#gym
#numpy
#pandas

import numpy as np
import pandas as pd
import tensorflow as tf

#deep Q-learning network :
#creating the class:
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            ):
        
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        #total learning step:
        self.learn_step_counter = 0

        #initialize zero memory :
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        #consist of :
        self.build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.sessoin()

        if output_graph():
            # tensorboard -- login dir = logs
            #tf.train.summaryWriter soon be deprecated, use following:
            tf.summary.FileWriter('logs/', self.sess.graph)

        self.sess.run(tf.global_variable_initializer())
        self.cost_his = []

    def _build_net(self):
        #-------build evaluate net ------
        # a input data:
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q-target')
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
            ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
            #configuration of layers
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            #first layer : collections is used later assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections = c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer : collections is used later assign to target net
            with tf.variable_scope('l1'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections = c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer = b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

    def store_transition(self, s,a,r,s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s,[a, r], s_))

        #replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
    
    def learn(self) :
        #check to replace target parameter :
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self, q_next, self, q_eval],
            feed_dict = {
                #fixed params
                self.s_: batch_memory[:, -self.n_features:], 
                #newest params
                self.s: batch_memory[:, :self.n_features],
            })
        
        #changing the q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        #context - 2 --> check README.md

        #train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={self.s: batch_memory[:, :self.n_features], self.q_target: q_target})

        self.cost_his.append(self.cost)

        #inreasing episilon
        self.eplison = self.eplison + self.splison_increment if self.eplison < self.eplison_max else self.eplison_max
        self.learn_step_counter += 1

    def plit_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arrange(len(self.cost_his)), self.cost_his)
        plt.ylabel('cost')
        plt.xlabel('training steps')
        plt.show()

    

        


