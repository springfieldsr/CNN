import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#extract test_batch file
def extracTestData():
    with open("test_batch", mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    
    data = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return data, labels

#extract batch_data file using batch_id
def extractBatchData(path, batch_id):
    with open(path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

#normalize data before fitting model
def normalize(x_train):
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    return (x_train-mean)/(std+1e-7)
    #min_val = np.min(x)
    #max_val = np.max(x)
    #x = (x-min_val) / (max_val-min_val)
    #return x

#cast original label list to one hot key encode
def oneHotEncode(x):
    encoded = np.zeros((len(x), 10))
    
    for i in range(len(x)):
        encoded[i][x[i]] = 1
    
    return encoded

#add more data to cross validation file
def preprocessBatchData(normalize, oneHotEncode, features, labels, filename):
    features = normalize(features)
    labels = oneHotEncode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

#create cross validation file
def crossValidationSet(filePath, normalize, oneHotEncode):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = extractBatchData(filePath, batch_i)
        index_of_validation = int(len(features) * 0.1)
        preprocessBatchData(normalize, oneHotEncode,
                             features[:-index_of_validation], labels[:-index_of_validation], 
                             'preprocess_batch_' + str(batch_i) + '.p')

        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    preprocessBatchData(normalize, oneHotEncode,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    with open(filePath + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    preprocessBatchData(normalize, oneHotEncode,
                         np.array(test_features), np.array(test_labels),
                         'preprocess_training.p')

def conv_net(x,keep_prob):
    l2 = []
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 32], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.08))
    conv5_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv6_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], mean=0, stddev=0.08))

    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1_l2 = 1e-4*tf.nn.l2_loss(conv1)
    l2.append(conv1_l2)
    conv1 = tf.nn.relu(conv1)
    #conv1_bn = tf.layers.batch_normalization(conv1)

    conv2 = tf.nn.conv2d(conv1, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2_l2 = 1e-4*tf.nn.l2_loss(conv2)
    l2.append(conv2_l2)
    conv2 = tf.nn.relu(conv2)
    #conv2_bn = tf.layers.batch_normalization(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv2_pool = tf.nn.dropout(conv2_pool, keep_prob)

    conv3 = tf.nn.conv2d(conv2_pool, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3_l2 = 1e-4*tf.nn.l2_loss(conv3)
    l2.append(conv3_l2)
    conv3 = tf.nn.relu(conv3)
    #conv3_bn = tf.layers.batch_normalization(conv3)


    conv4 = tf.nn.conv2d(conv3, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4_l2 = 1e-4*tf.nn.l2_loss(conv4)
    l2.append(conv4_l2)
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #conv4_bn = tf.layers.batch_normalization(conv4_pool)
    conv4_bn = tf.nn.dropout(conv4_pool, keep_prob)

    conv5 = tf.nn.conv2d(conv4_bn, conv5_filter, strides=[1,1,1,1], padding='SAME')
    conv5_l2 = 1e-4*tf.nn.l2_loss(conv5)
    l2.append(conv5_l2)
    conv5 = tf.nn.relu(conv5)
    #conv5_bn = tf.layers.batch_normalization(conv5)

    conv6 = tf.nn.conv2d(conv5, conv6_filter, strides=[1,1,1,1], padding='SAME')
    conv6_l2 = 1e-4*tf.nn.l2_loss(conv6)
    l2.append(conv6_l2)
    conv6 = tf.nn.relu(conv6)
    conv6_pool = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #conv6_bn = tf.layers.batch_normalization(conv6_pool)
    conv6_bn = tf.nn.dropout(conv6_pool, keep_prob)

    flat = tf.contrib.layers.flatten(conv6_bn)

    out = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=10, activation_fn=None)
    return out, l2


'''
def conv_net(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], mean=0, stddev=0.08))
    #conv5_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 512, 1024], mean=0, stddev=0.08))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    # 7, 8
    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)

    #conv5 = tf.nn.conv2d(conv4_bn, conv5_filter, strides=[1,1,1,1], padding='SAME')
    #conv5 = tf.nn.relu(conv5)
    #conv5_pool = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    #conv5_bn = tf.layers.batch_normalization(conv5_pool)

    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)

    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # 12
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalization(full3)

    # 13
    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    full4 = tf.nn.dropout(full4, keep_prob)
    full4 = tf.layers.batch_normalization(full4)

    # 14
    out = tf.contrib.layers.fully_connected(inputs=full4, num_outputs=10, activation_fn=None)
    return out
'''

#return batch features and labels
def returnBatch(features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

#load crossvalidation batch file
def loadPreprosessBatch(batch_id, batch_size):
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    return returnBatch(features, labels, batch_size)

def main():
    epochs = 100
    batch_size = 64
    keep_probability = 0.6
    crossValidationSet("./", normalize, oneHotEncode)
    valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
    tf.reset_default_graph()

    #learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(0.001, global_step, 1000, 0.9)

    # Inputs
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    l2 = []
    logits, l2 = conv_net(x, keep_prob)

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step) 

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    los = []
    acc = []
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in loadPreprosessBatch(batch_i, batch_size):
                    sess.run(optimizer, 
                    feed_dict={
                        x: batch_features,
                        y: batch_labels,
                        keep_prob: keep_probability 
                    })
                
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                loss = sess.run(cost, 
                        feed_dict={
                            x: batch_features,
                            y: batch_labels,
                            keep_prob: 1
                      })
                los.append(loss)
                valid_acc = sess.run(accuracy, 
                           feed_dict={
                                x: valid_features,
                                y: valid_labels,
                                keep_prob: 1
                           })
                acc.append(valid_acc)
                print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

        plt.yticks(np.arange(0, 2, step=0.1))
        plt.plot(acc, label='accuracy')
        plt.plot(los, label = 'loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy and Loss')
        plt.legend(loc='lower right')
        plt.savefig('./acc_loss.png')
        
        saver = tf.train.Saver()
        print(saver.save(sess, "./trainedmodel"))


main()