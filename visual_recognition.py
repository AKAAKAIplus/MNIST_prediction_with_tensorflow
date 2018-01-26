import numpy as np
# Import MINST data
import mnist_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = mnist_data.read_data_sets("mnist_data/", one_hot=True, reshape=False)

# understand the size (1 , 28 , 28)
batch_xs, batch_ys = mnist.train.next_batch(1)
print(len(batch_xs[0]))
print(len(batch_xs[0][0]))
print(len(batch_xs[0][0][0]))
print(batch_ys)

import tensorflow as tf
 
# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 128
display_step = 10
 
# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
 
def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

original_graph = tf.Graph()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_original = tf.Session(graph = original_graph)
with original_graph.as_default():
    
    obs_shape = mnist.train.get_observation_size()
    num_class = mnist.train.labels.shape[1]
    # tf Graph input

    x = tf.placeholder(tf.float32, shape=(None,) + obs_shape)
    y = tf.placeholder(tf.float32, (None, num_class))
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
     

    # conv1
    with tf.variable_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])# 5x5 conv, 1 input, 32 outputs
        b_conv1 = bias_variable([32])
        # print(conv2d(x, W_conv1))
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    # conv2
    with tf.variable_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])# 5x5 conv, 32 input, 64 outputs
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)


    # fc1
    with tf.variable_scope("fc1"):
        print(h_pool2.get_shape())
        # np.prod -> times every element in array
        shape = int(np.prod(h_pool2.get_shape()[1:]))
        W_fc1 = weight_variable([shape, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, shape])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fc2
    with tf.variable_scope("fc2"):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

     
    # Initializing the variables
    #init = tf.initialize_all_variables()
    init =tf.global_variables_initializer()

    saver = tf.train.Saver()

tf.reset_default_graph()

# Launch the graph
sess_original.run(init)
    

step = 1
# Keep training until reach max iterations
while step * batch_size < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # Fit training using batch data
    sess_original.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
    if step % display_step == 0:
        # Calculate batch accuracy
        acc = sess_original.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
        # Calculate batch loss
        print ("Iter " + str(step*batch_size) + ", Training Accuracy= " + "{:.5f}".format(acc))
    step += 1

print ("Optimization Finished!")
# Calculate accuracy for 256 mnist test images

acc = sess_original.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.})
print ("Testing Accuracy:" + "{:.5f}".format(acc))


with sess_original as sess_original:
    save_path = saver.save(sess_original, "model/model.ckpt")
    print("Model saved in file: %s" % save_path)



