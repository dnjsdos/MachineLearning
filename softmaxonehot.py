import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))
logits = tf.matmul(x, W) + b
y_pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        _, current_loss = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y})

    correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("정확도 : {}".format(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})))
