import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data", one_hot=True)

learning_late = 0.02
training_epochs = 50
batch_size = 256
display_step = 1
example_to_show = 10
input_size = 784
hidden1_size = 256
hidden2_size = 128

x = tf.placeholder(tf.float32, shape=[None, input_size])


def auto_encoder(x):
    w1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    h1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    w2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    h2 = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)
    w3 = tf.Variable(tf.random_normal(shape=[hidden2_size, hidden1_size]))
    b3 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    h3 = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)
    w4 = tf.Variable(tf.random_normal(shape=[hidden1_size, input_size]))
    b4 = tf.Variable(tf.random_normal(shape=[input_size]))
    h_out = tf.nn.sigmoid(tf.matmul(h3, w4) + b4)

    return h_out


reconstructed_x = auto_encoder(x)

loss = tf.reduce_mean(tf.square(reconstructed_x - x))
train_step = tf.train.RMSPropOptimizer(learning_rate=learning_late).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):

        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, _ = mnist.train.next_batch(batch_size)
            _, current_loss = sess.run([train_step, loss], feed_dict={x:batch_x})

        if epoch % display_step == 0:
            print("Epoch is {}, loss is {}.".format(epoch+1, current_loss))

    reconstructed_result = sess.run(reconstructed_x, feed_dict={x:mnist.test.images[:example_to_show]})

    f, a = plt.subplots(2, example_to_show, figsize=(example_to_show, 2))

    for i in range(example_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(reconstructed_result[i], (28, 28)))

    f.savefig('./reconstructed_mnist_image.png')
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
