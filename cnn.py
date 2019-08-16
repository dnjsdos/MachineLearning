import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data", one_hot=True)


def build_cnn_classifier(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 5*5 kernel, 32 filters
    # 28*28*1 -> 28*28*32 by convolution
    w_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

    # 28*28*32 -> 14*14*32 by max pooling
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 5*5 kernel, 64 filters
    # 14*14*32 -> 14*14*64
    w_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

    # 14*14*64 -> 7*7*64 by max pooling
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    h_pool2_flat = tf.reshape(h_pool2, shape=[-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    w_out = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=5e-2))
    b_out = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1, w_out) + b_out
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits


x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

y_pred, logits = build_cnn_classifier(x)

#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred)))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch_x, batch_y = mnist.train.next_batch(50)

        sess.run(train_step, feed_dict={x:batch_x, y:batch_y})

        if i % 100 == 0:
            training_accuracy = accuracy.eval(feed_dict={x:batch_x, y:batch_y})
            print("Repeated Epoch : {}, Training data accuracy : {}".format(i, training_accuracy))

    print("Training was over.\nTesting Accuracy is {}".format(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})))
