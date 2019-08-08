import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data", one_hot=True)

learning_rate = 0.01
num_epochs = 30
batch_size = 256
display_step = 1
input_size = 784
hidden1_size = 256
hidden2_size = 256
output_size = 10

x = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, output_size])


def build_ANN(x):
    W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    H1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    H2 = tf.nn.sigmoid(tf.matmul(H1, W2) + b2)
    W_out = tf.Variable(tf.random_normal(shape=[hidden2_size, output_size]))
    b_out = tf.Variable(tf.random_normal(shape=[output_size]))
    logits = tf.matmul(H2, W_out) + b_out

    return logits


logits = build_ANN(x)
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

'''
GradientDescentOptimizer : 76%
AdagradOptimizer : 84.4%
RMSPropOptimizer : 96.35%
AdamOptimizer : 96.27%

tf.nn.softmax_cross_entropy_with_logits -> tf.nn.sigmoid_cross_entropy_with_logits : 1% accuracy up
'''

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)
        average_loss = 0.

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, current_loss = sess.run([train_step, loss], feed_dict={xg:batch_x, y:batch_y})
            average_loss += current_loss / total_batch

        print("Epoch loss : {}".format(average_loss))

    correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy : {}".format(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})))

