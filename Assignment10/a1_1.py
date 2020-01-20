import tensorflow.compat.v1 as tf  # remove compat.v1 if tf version 1
tf.disable_v2_behavior()  # disable if tf version 1


def main():
    epochs = 30
    # a) i
    a = tf.constant(2, dtype=tf.float32)
    b = tf.constant(4, dtype=tf.float32)
    x = tf.Variable(0, dtype=tf.float32)
    f = a * tf.pow(tf.add(x, 1.0), 2) + b * x

    # a) ii
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(f)

    # b)
    with tf.Session() as sess:
        # i
        tf.global_variables_initializer().run()

        # ii
        for i in range(epochs):
            sess.run(optimizer)
            f_val = sess.run(f)
            x_val = sess.run(x)
            print("Epoch %d: f(x) = %f; x = %f" % (i, f_val, x_val))

        x_opt = x_val
        function_opt = f_val


if __name__ == '__main__':
    main()
