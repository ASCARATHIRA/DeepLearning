import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib import rnn
from scipy import ndimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

log_dir = Path("./logs_LSTM32")
if not log_dir.exists():
	subprocess.call(["./download.sh"])

#user imports
import data_loader
from CellDefinitions import MyLSTMCell, MyGRUCell

import argparse

parser = argparse.ArgumentParser(description='Specify model parameters and function')
parser.add_argument('--model', dest="model", type=str, help="Model type: lstm/gru")
parser.add_argument('--hidden_unit', dest="hidden_dim", type=int, choices = [32, 64, 128, 256], help="Number of hidden units (dimension of hidden layer)")
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()


celltype = args.model


tf.set_random_seed(123)  # reproducibility

dl = data_loader.DataLoader()
train_data, train_labels = dl.load_data('train')
train_labels = np.eye(10)[np.asarray(train_labels, dtype=np.int32)]
test_data, test_labels = dl.load_data('test')
test_labels = np.eye(10)[np.asarray(test_labels, dtype=np.int32)]

train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=123)

learning_rate = 0.01
training_epochs = 50
batch_size = 100

learning_rate = 0.001
training_epochs = 50
batch_size = 100
num_hidden = args.hidden_dim
timesteps = 28
num_classes = 10



class RNN(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            self.mode = tf.placeholder(tf.bool)
            self.keep_prob = tf.placeholder(tf.float32)
            self.input_layer = tf.reshape(self.X, [-1, 28, 28])
            
            self.w_out = tf.Variable(tf.random_normal([num_hidden, num_classes]))
            self.b_out = tf.Variable(tf.random_normal([num_classes]))
            
            
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, timesteps, n_input)
            # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
            self.x = tf.unstack(self.input_layer, timesteps, 1)
            if celltype == "lstm":
                rnn_cell = MyLSTMCell(num_hidden)
            elif celltype == "gru":
                rnn_cell = MyGRUCell(num_hidden)
            
            
            outputs, states = rnn.static_rnn(rnn_cell, self.x, dtype=tf.float32)
            self.logits = tf.matmul(outputs[-1], self.w_out) + self.b_out
            #self.matrix = rnn_cell.matrix

            self.pred = tf.nn.softmax(self.logits)
            # Test model and check accuracy
            self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)
            # define cost/loss & optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.summary.scalar('mean_loss', self.cost)
            self.merged = tf.summary.merge_all()

            # When using the batchnormalization layers,
            # it is necessary to manually add the update operations
            # because the moving averages are not included in the graph            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):                     
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost, global_step=self.global_step)

nn = RNN()
# Best validation accuracy seen so far.
best_validation_accuracy = 0.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvement found in this many iterations.
patience = 5

# Start session
if celltype == "lstm":
	LOGDIR = 'logs_LSTM'+str(num_hidden)
elif celltype == "gru":
	LOGDIR = 'logs_GRU'+str(num_hidden)


if args.train:
    sv = tf.train.Supervisor(graph=nn.graph,
                         logdir=LOGDIR,
                         summary_op=None,
                         save_model_secs=0)

    with sv.managed_session(config=tf.ConfigProto(device_count={'GPU':1})) as sess:
        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = int(len(train_data) / batch_size)
            if sv.should_stop(): break
            for i in range(total_batch):
                batch_xs, batch_ys = train_data[(i)*batch_size:(i+1)*batch_size], train_labels[(i)*batch_size:(i+1)*batch_size]
                feed_dict = {nn.X: batch_xs, nn.Y: batch_ys, nn.mode:True, nn.keep_prob:0.8}
                c, _ = sess.run([nn.cost, nn.optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch
                if i%50:
                
                    sv.summary_computed(sess, sess.run(nn.merged, feed_dict))
                    gs = sess.run(nn.global_step, feed_dict)
        
            print 'Epoch : ' + str(epoch) + ' Training Loss: ' + str(avg_cost)
            acc = sess.run(nn.accuracy, feed_dict={
                        nn.X: val_data, nn.Y: val_labels, nn.mode:False, nn.keep_prob:1.0})
            print 'Validation Accuracy: ' + str(acc)
            if acc > best_validation_accuracy:
                last_improvement = epoch
                best_validation_accuracy = acc
                sv.saver.save(sess, LOGDIR + '/model_gs', global_step=gs)
            if epoch - last_improvement > patience:
                print("Early stopping ...")
                break
            

if args.test:
    nn = RNN()
    print("Graph loaded")
    with nn.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(LOGDIR))
            print("Restored!")
            acc = sess.run(nn.accuracy, feed_dict={
                  nn.X: test_data, nn.Y: test_labels, nn.mode:False, nn.keep_prob:1.0})
            print('Accuracy:', acc)
        


