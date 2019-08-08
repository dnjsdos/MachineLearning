import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/", one_hot=False)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.int64, shape=[None])

W = tf.Variable(tf.random_normal(shape=[784, 10]))
b = tf.Variable(tf.random_normal(shape=[10]))

logits = tf.matmul(x, W) + b
y_pred = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
train_step = tf.train.AdagradOptimizer(learning_rate=0.5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        _, current_loss = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y})

        if i % 10 == 0:
            print("현재 loss는 {} 입니다".format(current_loss))

    correct_prediction = tf.equal(tf.math.argmax(y_pred, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("정확도 : {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))