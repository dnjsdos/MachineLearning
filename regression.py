import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal(shape=[1]))
b = tf.Variable(tf.zeros(shape=[1]))

linear_model = W*x + b

loss = tf.reduce_mean(tf.square(y - linear_model))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

train_x = [1, 2, 3, 4]
train_y = [3, 5, 7, 9]

test_x = [1.5, 7, 8, 10]
test_y = [4, 15, 17, 21]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        _, currentloss = sess.run([train_step, loss], feed_dict={x:train_x, y:train_y})

        if i % 10 == 0:
            print("{}번째 트레이닝 중 loss는 {}입니다".format(i, currentloss))

    print(sess.run(linear_model, feed_dict={x:test_x, y:test_y}))