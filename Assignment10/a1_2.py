import tensorflow.compat.v1 as tf  # remove compat.v1 if tf version 1
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()  # disable if tf version 1


def main():
    # a)
    a = tf.constant(2, dtype=tf.float32)
    b = tf.constant(4, dtype=tf.float32)
    x = tf.placeholder(dtype=tf.float32)
    f = a * tf.pow(tf.add(x, 1.0), 2) + b * x

    # b)
    with tf.Session() as sess:
        x_range = np.arange(-6, 2, 8 / 50)
        function_range = sess.run(f, feed_dict={x: x_range})
        plt.figure()
        plt.plot(x_range, function_range)
        plt.plot((-2, -2), (-7, 25))
        plt.legend(["f(x)", "minimum"])
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Plot of f(x) and minimum")
        plt.savefig("a1_2.eps")
        plt.show()


if __name__ == '__main__':
    main()
