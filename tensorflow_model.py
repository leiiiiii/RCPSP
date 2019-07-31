#Neural Network Model using package tensorflow
import tensorflow as tf

def createNeuralNetworkModel(input_size, output_size, learningRate):
    # setup placeholder
    Input = tf.placeholder(tf.float32, shape=[None, input_size],name='Input')
    Output = tf.placeholder(tf.float32, shape=[None, output_size],name='Output')

    # define variables(tensors)
    W1 = tf.get_variable('w1', [input_size, 128], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [128], initializer=tf.zeros_initializer())

    W2 = tf.get_variable('w2', [128, 256], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [256], initializer=tf.zeros_initializer())

    W3 = tf.get_variable('w3', [256, output_size], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [output_size], initializer=tf.zeros_initializer())

    # logits(Z)
    O1 = tf.nn.sigmoid(tf.matmul(Input, W1) + b1, name='O1')
    # O1 = tf.layers.dropout(O1, rate=0.2)  # dropout
    O2 = tf.nn.sigmoid(tf.matmul(O1, W2) + b2, name='O2')
    # O2 = tf.layers.dropout(O2, rate=0.2)  # dropout
    Z3 = tf.matmul(O2, W3) + b3  # not activated

    # cost(calculate cost has special function,Z3 don't have to activated
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Output))

    # # compute accuracy
    correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Output))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    return train_step,accuracy






