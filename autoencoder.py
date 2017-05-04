import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import function

def gen_data(n, block_len):
    return np.random.randint(0, 2, size=(n, block_len))*2-1

@function.Defun()
def binarize_grad(x, dy):
    return dy

@function.Defun(grad_func=binarize_grad)
def binarize(x):
    return tf.floor(x)*2+1

learning_rate = 0.01
num_epochs = 20
batch_size = 256

n_hidden_1 = 8
n_hidden_2 = 16
n_input = 4

x = tf.placeholder("float", [None, n_input])

weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
        }
biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input]))
        }

def encoder(x):
    layer1 = tf.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer2 = tf.tanh(tf.add(tf.matmul(layer1, weights['encoder_h2']), biases['encoder_b2']))
    return layer2

def decoder(x):
    layer1 = tf.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer2 = tf.tanh(tf.add(tf.matmul(layer1, weights['decoder_h2']), biases['decoder_b2']))
    return layer2

encoder_op = encoder(x)
decoder_op = binarize(decoder(binarize(encoder_op)))

y_pred = decoder_op
y_true = x

cost = tf.reduce_mean(tf.pow(y_pred-y_true, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    total_batch = 500
    for epoch in range(num_epochs):
        for i in range(total_batch):
            batch_x = gen_data(batch_size, n_input)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x})
        print('Epoch:', str(epoch+1), 'cost:', str(c))

    actual, encode_decode = sess.run([y_true, y_pred], feed_dict={x: gen_data(10, n_input)})
    for i in range(10):
        print(actual[i], encode_decode[i])
