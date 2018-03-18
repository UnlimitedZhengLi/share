from functools import reduce
import tensorflow as tf
import numpy as np


class RMSPropOptimizer:
    def __init__(self, learning_rate, rho=0.9, delta=1e-6):
        """learning_rate is a tensor."""

        with tf.name_scope('Optimizer') as self.Optimizer:
            self.learning_rate = learning_rate
            self.rho = tf.constant(rho, dtype=tf.float32, name='rho')
            self.delta = tf.constant(delta, dtype=tf.float32, name='delta')

            self.r = tf.Variable(0, trainable=False, dtype=tf.float32, name='r')

    def minimize(self, loss):
        theta = tf.trainable_variables()
        grad = tf.gradients(loss, theta)
        grad_norm = reduce(lambda x, y: tf.add(tf.reduce_sum(tf.multiply(x, x)),
                                               tf.reduce_sum(tf.multiply(y, y))), grad)

        self.r.assign(value=tf.add(tf.multiply(self.rho, self.r),
                                   tf.multiply(1 - self.rho, grad_norm)), use_locking=False)

        ops = []
        for var, value in zip(theta, grad):
            sub = tf.multiply(tf.div(self.learning_rate, tf.sqrt(tf.add(self.delta, self.r))), value)
            ops.append(var.assign_sub(sub))

        return ops

    @classmethod
    def unit_test(cls):
        """RMSPropOptimizer.unit_test()"""
        with tf.name_scope('neural_network'):
            X = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='X')
            Y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='Y')
            lr = tf.placeholder(dtype=tf.float32, name='lr')

            W = tf.Variable(initial_value=tf.truncated_normal(shape=[3, 2], stddev=0.1), dtype=tf.float32, name='W')
            b = tf.Variable(initial_value=tf.constant([0, 0], dtype=tf.float32), name='b')

            Y_ = tf.nn.relu(tf.add(tf.matmul(X, W), b), name='Y_')
            cost = tf.reduce_mean(tf.square(tf.subtract(Y_, Y)), name='cost')

            train_step_1 = RMSPropOptimizer(learning_rate=lr).minimize(cost)
            train_step_2 = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)

        with tf.Session() as sess_1:
            data_X = np.random.rand(100, 3)
            data_Y = np.random.rand(100, 2)
            init = tf.global_variables_initializer()
            with tf.Session() as sess_2:
                sess_1.run(init)
                sess_2.run(init)

                for i in range(500):
                    c_1 = sess_1.run(cost, feed_dict={X: data_X, Y: data_Y})
                    c_2 = sess_2.run(cost, feed_dict={X: data_X, Y: data_Y})
                    print(i, ':', 'my_cost = %.8f' % c_1, '\ttf_cost = %.8f' % c_2)

                    sess_1.run(train_step_1, feed_dict={X: data_X, Y: data_Y, lr: 0.00001})
                    sess_2.run(train_step_2, feed_dict={X: data_X, Y: data_Y, lr: 0.00001})


class NesterovRMSPropOptimizer(RMSPropOptimizer):
    def __init__(self, alpha=0.9, *args, **kwargs):
        super(NesterovRMSPropOptimizer, self).__init__(*args, **kwargs)

        with tf.name_scope('Optimizer') as self.Optimizer:
            self.alpha = tf.constant(alpha, dtype=tf.float32, name='alpha')
            self.v = [tf.Variable(tf.zeros_like(var), trainable=False, dtype=tf.float32)
                      for var in tf.trainable_variables()]

    def minimize(self, loss):
        # script 0，update theta.
        # I must return this op. if not, the trainable_variables will not update in the first time.
        theta = tf.trainable_variables()
        update_theta_0 = [var_1.assign_add(tf.multiply(self.alpha, var_2)) for var_1, var_2 in zip(theta, self.v)]

        # script 1，update theta.
        grad = tf.gradients(loss, theta)
        update_theta_1 = [var_1.assign_sub(tf.multiply(self.alpha, var_2)) for var_1, var_2 in zip(theta, self.v)]

        # script 2，update r
        grad_norm = reduce(lambda x, y: tf.add(tf.reduce_sum(tf.multiply(x, x)),
                                               tf.reduce_sum(tf.multiply(y, y))), grad)
        update_r = self.r.assign(value=tf.add(tf.multiply(self.rho, self.r),
                                              tf.multiply(1 - self.rho, grad_norm)), use_locking=False)

        # script 3, update v
        update_v = []
        for var, value in zip(self.v, grad):
            assign_op = tf.subtract(tf.multiply(self.alpha, var),
                                    tf.multiply(tf.div(self.learning_rate, tf.sqrt(update_r)), value))
            update_v.append(var.assign(assign_op))

        # script 4, update theta
        update_theta_2 = []
        for var, value in zip(theta, self.v):
            update_theta_2.append(var.assign_add(value))

        return update_theta_0, update_theta_1, update_r, update_v, update_theta_2

    @classmethod
    def unit_test(cls):
        """RMSPropOptimizer.unit_test()"""
        with tf.name_scope('neural_network'):
            X = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='X')
            Y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='Y')
            lr = tf.placeholder(dtype=tf.float32, name='lr')

            W = tf.Variable(initial_value=tf.truncated_normal(shape=[3, 2], stddev=0.1), dtype=tf.float32, name='W')
            b = tf.Variable(initial_value=tf.constant([0, 0], dtype=tf.float32), name='b')

            Y_ = tf.nn.relu(tf.add(tf.matmul(X, W), b), name='Y_')
            cost = tf.reduce_mean(tf.square(tf.subtract(Y_, Y)), name='cost')

            train_step_1 = NesterovRMSPropOptimizer(learning_rate=lr).minimize(cost)
            train_step_2 = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(cost)

        with tf.Session() as sess_1:
            data_X = np.random.rand(100, 3)
            data_Y = np.sin(data_X[:, :2]) + np.cos(data_X[:, 1:])
            init = tf.global_variables_initializer()
            with tf.Session() as sess_2:
                sess_1.run(init)
                sess_2.run(init)

                for i in range(500):
                    c_1 = sess_1.run(cost, feed_dict={X: data_X, Y: data_Y})
                    c_2 = sess_2.run(cost, feed_dict={X: data_X, Y: data_Y})
                    print(i, ':\n', 'my_cost = %.8f' % c_1, '\ttf_cost = %.8f' % c_2)

                    sess_1.run(train_step_1, feed_dict={X: data_X, Y: data_Y, lr: 0.001})
                    sess_2.run(train_step_2, feed_dict={X: data_X, Y: data_Y, lr: 0.001})


with tf.Graph().as_default():
    RMSPropOptimizer.unit_test()

with tf.Graph().as_default():
    NesterovRMSPropOptimizer.unit_test()


# X = tf.Variable([1, 2], dtype=tf.float32, name='X')
#
#
# def foo():
#     add_ops = X.assign_add(tf.constant([1, 1], dtype=tf.float32))
#     sub_ops = X.assign_sub(tf.constant([1, 0], dtype=tf.float32))
#     return add_ops, sub_ops  # 虽然一般来说在输出一个函数时，是不改变变量的，但是如果对函数内部的变量重复进行了assign操作，则执行顺序是按照列表内部的iter的顺序进行的。
#
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     sess.run(foo())
#     print(sess.run(X))
