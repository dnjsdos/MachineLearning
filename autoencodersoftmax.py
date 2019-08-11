import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=True)

learning_rate_RMSProp = 0.02
learning_rate_gradientdescent = 0.5
num_epochs = 100
batch_size = 256
display_step = 1
input_size = 784
hidden1_size = 128
hidden2_size = 64
output_size = 10

x = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, output_size])


def build_autoencoder(x):
    w1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

    w2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)

    w3 = tf.Variable(tf.random_normal(shape=[hidden2_size, hidden1_size]))
    b3 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    h3 = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)

    w_out = tf.Variable(tf.random_normal(shape=[hidden1_size, input_size]))
    b_out = tf.Variable(tf.random_normal(shape=[input_size]))
    X_reconstructed = tf.nn.sigmoid(tf.matmul(h3, w_out) + b_out)

    return X_reconstructed, h2


def build_softmax_classifier(x):
    w_softmax = tf.Variable(tf.random_normal(shape=[hidden2_size, output_size]))
    b_softmax = tf.Variable(tf.random_normal(shape=[output_size]))
    y_pred = tf.nn.softmax(tf.matmul(x, w_softmax) + b_softmax)

    return y_pred

y_pred, extracted_features = build_autoencoder(x)
y_true = x

y_pred_softmax = build_softmax_classifier(extracted_features)

pretraining_loss = tf.reduce_mean(tf.square(y_true - y_pred))
pretraining_train_step = tf.train.RMSPropOptimizer(learning_rate_RMSProp).minimize(pretraining_loss)

finetuning_loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred_softmax), reduction_indices=[1]))
finetuning_train_step = tf.train.AdamOptimizer(0.01).minimize(finetuning_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(num_epochs):

        for i in range(total_batch):

            batch_x, _ = mnist.train.next_batch(batch_size)
            _, current_pretraining_loss = sess.run([pretraining_train_step, pretraining_loss], feed_dict={x:batch_x})

        if epoch % display_step == 0:
            print("Epoch is {}, Pretraning loss is {}".format(epoch+1, current_pretraining_loss))

    print("Step 1 : Pre-training for mnist data reconstruction has been completed.")

    for epoch in range(num_epochs):

        for i in range(total_batch):

            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, current_finetuning_loss = sess.run([finetuning_train_step, finetuning_loss], feed_dict={x:batch_x, y:batch_y})

        if epoch % display_step == 0:
            print("Epoch is {}, Pretraning loss is {}".format(epoch+1, current_finetuning_loss))

    print("Step 2 : Autoencoder + softmax classifier optimization which was made for mnist data classifying has been completed")

    correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(y_pred_softmax, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy is {}".format(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})))