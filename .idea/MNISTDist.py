from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                           "Directory for string mnist data")
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_integer("hidden_units", 100,
                            "Number of units in the hidden layer of the NN")
tf.app.flags.DEFINE_integer("batch_size",128,"Training batchsize")
tf.app.flags.DEFINE_integer("training_iter",10000, "Training iteration")
tf.app.flags.DEFINE_float("learning_rate",0.001, "Learning rate")
tf.app.flags.DEFINE_integer("display_step",100, "display step")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the   job")

NUM_INPUT = 784
NUM_CLASSES = 10
DROPOUT = 0.75

FLAGS = tf.app.flags.FLAGS
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

            x = tf.placeholder(tf.float32, [None, NUM_INPUT])
            y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
            keep_prob = tf.placeholder(tf.float32)  # dropout

            weights = {
                # 5x5 conv, 1 input, 32 outputs
                #'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
                'wc1': weight_variable([5, 5, 1, 32]),
                # 5x5 conv, 32 inputs, 64 outputs
                #'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
                'wc2': weight_variable([5, 5, 32, 64]),
                # fully connected, 7*7*64 inputs, 1024 outputs
                #'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
                'wd1': weight_variable([7 * 7 * 64, 1024]),
                # 1024 inputs, 10 outputs (class prediction)
                #'out': tf.Variable(tf.random_normal([1024, NUM_CLASSES]))
                'out': weight_variable([1024, NUM_CLASSES])
            }

            biases = {
                #'bc1': tf.Variable(tf.random_normal([32])),
                'bc1': bias_variable([32]),
                #'bc2': tf.Variable(tf.random_normal([64])),
                'bc2': bias_variable([64]),
                #'bd1': tf.Variable(tf.random_normal([1024])),
                'bd1': bias_variable([1024]),
                #'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
                'out': bias_variable([NUM_CLASSES])
            }

            # Construct model
            pred = conv_net(x, weights, biases, keep_prob)

            # Define loss and optimizer
            global_step = tf.Variable(0)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost, global_step=global_step)

            # Evaluate model
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            saver = tf.train.Saver()
            summary_op = tf.merge_all_summaries()
            init_op = tf.initialize_all_variables()

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/tmp/train_logs",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

        # The supervisor takes care of session initialization, restoring from
        with sv.managed_session(server.target) as sess:
            # Loop until the supervisor shuts down or steps have completed.
            step = 0
            while not sv.should_stop() and step < FLAGS.training_iter:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                train_feed = {x: batch_xs, y_: batch_ys, keep_prob: 1.}

                if step % FLAGS.display_step == 0:
                    loss, acc = sess.run([cost, accuracy],train_feed)
                    print("job: %s/%s" % (FLAGS.job_name, FLAGS.task_index),
                          "step: ", step,
                          "mini_batch loss: ", loss,
                          "training accuracy: ", acc)

                _, step = sess.run([optimizer, global_step], feed_dict=train_feed)

    # Ask for all the services to stop.
        # a checkpoint, and closing when done or an error occurs.
    sv.stop()
    print("Optimization Finished!")



if __name__ == "__main__":
    tf.app.run()