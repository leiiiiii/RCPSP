#Neural Network Model using package tensorflow
import tensorflow as tf

def createNeuralNetworkModel(input_size, output_size, learningRate):
    # setup placeholder
    Input = tf.placeholder(tf.float32, shape=[None, input_size],name='Input')
    Output = tf.placeholder(tf.float32, shape=[None, output_size],name='Output')

    # define variables(tensors)
    W1 = tf.get_variable('w1', [input_size, 128], initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram("weight_1", W1)
    b1 = tf.get_variable('b1', [128],initializer=tf.constant_initializer(value=0.3))
    tf.summary.histogram("bias_1", b1)

    W2 = tf.get_variable('w2', [128, 256], initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram("weight_2", W2)
    b2 = tf.get_variable('b2',[256] , initializer=tf.constant_initializer(value=0.15))
    tf.summary.histogram("bias_2", b2)

    W3 = tf.get_variable('w3', [256, output_size], initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram("weight_3", W3)
    b3 = tf.get_variable('b3', [output_size], initializer=tf.zeros_initializer())
    tf.summary.histogram("bias_3", b3)

    # logits(Z)
    O1 = tf.nn.sigmoid(tf.matmul(Input, W1) + b1, name='O1')
    tf.summary.histogram("O1", O1)
    # O1 = tf.layers.dropout(O1, rate=0.2)  # dropout

    O2 = tf.nn.sigmoid(tf.matmul(O1, W2) + b2, name='O2')
    tf.summary.histogram("O2", O2)
    # O2 = tf.layers.dropout(O2, rate=0.2)  # dropout

    with tf.name_scope("Z3"):
        Z3 = tf.matmul(O2, W3) + b3  # not activated
        tf.summary.histogram("Z3",Z3)
        tf.add_to_collection('pred_network',Z3)

    # cost(calculate cost has special function,Z3 don't have to activated
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Output))
        tf.summary.scalar('cost', cost)

    # # compute accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Output))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope("train_step"):
        train_step = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    return train_step,accuracy






