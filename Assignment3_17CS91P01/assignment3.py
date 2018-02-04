import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import data_loader
import sys
#%matplotlib inline


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

class NN(object):
    def __init__(self):
        
        if True:
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            self.Y = tf.placeholder(tf.float32, [None, 10])
            self.mode = tf.placeholder(tf.bool)
            self.keep_prob = tf.placeholder(tf.float32)
            
            self.b1 = tf.Variable(tf.zeros((100,)))
            self.W1 = tf.Variable(tf.random_uniform((784, 100), -1, 1))
            self.h1 = tf.matmul(self.X, self.W1) + self.b1
            self.h1 = tf.maximum(self.h1, tf.zeros(self.b1.shape))
            
            self.b2 = tf.Variable(tf.zeros((100,)))
            self.W2 = tf.Variable(tf.random_uniform((100, 100), -1, 1))
            self.h2 = tf.maximum(tf.matmul(self.h1, self.W2) + self.b2, tf.zeros(self.b2.shape))
            
            self.b3 = tf.Variable(tf.zeros((100,)))
            self.W3 = tf.Variable(tf.random_uniform((100, 100), -1, 1))
            self.h3 = tf.maximum(tf.matmul(self.h2, self.W3) + self.b3, tf.zeros(self.b3.shape))
            
            
            self.b4 = tf.Variable(tf.zeros((10,)))
            self.W4 = tf.Variable(tf.random_uniform((100, 10), -1, 1))
            self.logits = tf.matmul(self.h3, self.W4) + self.b4
            
            self.logits = self.logits/tf.reduce_mean(self.logits)
            self.logits = tf.exp(self.logits)
            self.pred = self.logits/tf.reduce_mean(self.logits)

            
            # Test model and check accuracy
            self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)
            
            # define cost/loss & optimizer
            self.cost = -tf.reduce_sum(self.Y * tf.log(self.pred))
            
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.summary.scalar('mean_loss', self.cost)
            self.merged = tf.summary.merge_all()

                                
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost, global_step=self.global_step)

        

# Best validation accuracy seen so far.
best_validation_accuracy = 0.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvement found in this many iterations.
patience = 10


# Start session

if len(sys.argv)==2 and sys.argv[1] == "--train":
    sess=tf.Session()
    nn = NN()
    sess.run(tf.global_variables_initializer())
    sv = tf.train.Supervisor(
                         logdir='weights/',
                         summary_op=None,
                         save_model_secs=0)

    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(train_data) / batch_size)
        if sv.should_stop(): break
        
        for i in range(total_batch):
            batch_xs, batch_ys = train_data[(i)*batch_size:(i+1)*batch_size], train_labels[(i)*batch_size:(i+1)*batch_size]
            feed_dict = {nn.X: batch_xs, nn.Y: batch_ys, nn.mode:True, nn.keep_prob:0.8}
            W1, pred, c, _ = sess.run([nn.W1, nn.pred, nn.cost, nn.optimizer], feed_dict=feed_dict)
            
            avg_cost += c / total_batch
            
            if i%50==0:
                sv.summary_computed(sess, sess.run(nn.merged, feed_dict))
                gs = sess.run(nn.global_step, feed_dict)
        
        print 'Epoch : ' + str(epoch) + ' Training Loss: ' + str(avg_cost)
        acc = sess.run(nn.accuracy, feed_dict={
                        nn.X: val_data, nn.Y: val_labels, nn.mode:False, nn.keep_prob:1.0})
        print 'Validation Accuracy: ' + str(acc)
        if acc > best_validation_accuracy:
            last_improvement = epoch
            best_validation_accuracy = acc
            sv.saver.save(sess, 'weights' + '/model_gs', global_step=gs)
        if epoch - last_improvement > patience:
            print("Early stopping ...")
            break
    sess.close()
            


if len(sys.argv)==2 and sys.argv[1] == "--test":
	# Load graph
    nn = NN()
    print("Graph loaded")

    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ## Restore parameters
        sv.saver.restore(sess, tf.train.latest_checkpoint('weights/'))
        print("Restored!")
        acc = sess.run(nn.accuracy, feed_dict={
              nn.X: test_data, nn.Y: test_labels, nn.mode:False, nn.keep_prob:1.0})
        print('Test Accuracy:', acc)
        

if len(sys.argv)==2 and sys.argv[1] == "--layer=1":
    clf_l2_LR = LogisticRegression(C=1, penalty='l2', tol=0.01)
    # Load graph
    nn = NN()
    print("Graph loaded")

    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ## Restore parameters
        sv.saver.restore(sess, tf.train.latest_checkpoint('weights/'))
        print("Restored!")
        trn_logits=[]
        trn_labels = []
        for i in range(len(train_data)):
            h1=sess.run(nn.h1, feed_dict={
                nn.X: train_data[i].reshape((1,784)), nn.Y: train_labels[i].reshape((1,10)), nn.mode:False, nn.keep_prob:1.0})
            trn_logits.append(np.argmax(h1))
            trn_labels.append(np.argmax(train_labels[i]))
        clf_l2_LR.fit(np.array(trn_logits).reshape(-1,1), trn_labels)
        tst_logits = []
        tst_labels = []
        for i in range(len(test_data)):
            h1=sess.run(nn.h1, feed_dict={
                nn.X: test_data[i].reshape((1,784)), nn.Y: test_labels[i].reshape((1,10)), nn.mode:False, nn.keep_prob:1.0})
            tst_logits.append(np.argmax(h1))
            tst_labels.append(np.argmax(test_labels[i]))
        pred_lbls = clf_l2_LR.predict(np.array(tst_logits).reshape(-1,1))
        acc = 0.0
        for i in range(len(pred_lbls)):
            p = pred_lbls[i]
            q = tst_labels[i]
            if p==q:
                acc += 1.0/len(pred_lbls)
        print "Test Accuracy using Logistic Regression with 1st hidden layer: ", acc		
        
if len(sys.argv)==2 and sys.argv[1] == "--layer=2":
    clf_l2_LR = LogisticRegression(C=1, penalty='l2', tol=0.01)
    # Load graph
    nn = NN()
    print("Graph loaded")

    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ## Restore parameters
        sv.saver.restore(sess, tf.train.latest_checkpoint('weights/'))
        print("Restored!")
        trn_logits=[]
        trn_labels = []
        for i in range(len(train_data)):
            h2=sess.run(nn.h2, feed_dict={
                nn.X: train_data[i].reshape((1,784)), nn.Y: train_labels[i].reshape((1,10)), nn.mode:False, nn.keep_prob:1.0})
            trn_logits.append(np.argmax(h2))
            trn_labels.append(np.argmax(train_labels[i]))
        clf_l2_LR.fit(np.array(trn_logits).reshape(-1,1), trn_labels)
        tst_logits = []
        tst_labels = []
        for i in range(len(test_data)):
            h2=sess.run(nn.h2, feed_dict={
                nn.X: test_data[i].reshape((1,784)), nn.Y: test_labels[i].reshape((1,10)), nn.mode:False, nn.keep_prob:1.0})
            tst_logits.append(np.argmax(h2))
            tst_labels.append(np.argmax(test_labels[i]))
        pred_lbls = clf_l2_LR.predict(np.array(tst_logits).reshape(-1,1))
        acc = 0.0
        for i in range(len(pred_lbls)):
            p = pred_lbls[i]
            q = tst_labels[i]
            if p==q:
                acc += 1.0/len(pred_lbls)
        print "Test Accuracy using Logistic Regression with 2nd hidden layer: ", acc			 
        
	
if len(sys.argv)==2 and sys.argv[1] == "--layer=3":
    clf_l2_LR = LogisticRegression(C=1, penalty='l2', tol=0.01)
    # Load graph
    nn = NN()
    print("Graph loaded")

    sv = tf.train.Supervisor()
    with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ## Restore parameters
        sv.saver.restore(sess, tf.train.latest_checkpoint('weights/'))
        print("Restored!")
        trn_logits=[]
        trn_labels = []
        for i in range(len(train_data)):
            h3=sess.run(nn.h3, feed_dict={
                nn.X: train_data[i].reshape((1,784)), nn.Y: train_labels[i].reshape((1,10)), nn.mode:False, nn.keep_prob:1.0})
            trn_logits.append(np.argmax(h3))
            trn_labels.append(np.argmax(train_labels[i]))
        clf_l2_LR.fit(np.array(trn_logits).reshape(-1,1), trn_labels)
        tst_logits = []
        tst_labels = []
        for i in range(len(test_data)):
            h3=sess.run(nn.h3, feed_dict={
                nn.X: test_data[i].reshape((1,784)), nn.Y: test_labels[i].reshape((1,10)), nn.mode:False, nn.keep_prob:1.0})
            tst_logits.append(np.argmax(h3))
            tst_labels.append(np.argmax(test_labels[i]))
        pred_lbls = clf_l2_LR.predict(np.array(tst_logits).reshape(-1,1))
        acc = 0.0
        for i in range(len(pred_lbls)):
            p = pred_lbls[i]
            q = tst_labels[i]
            if p==q:
                acc += 1.0/len(pred_lbls)
        print "Test Accuracy using Logistic Regression with 3rd hidden layer: ", acc
