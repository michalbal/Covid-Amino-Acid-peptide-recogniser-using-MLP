import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import *
import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

PEPTIDS_TO_RETURN = 5
PEPTIDE_LENGTH = 9

amino_acids_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
                    'P': 12, 'Q': 13, 'R': 14, 'S': 15,
                    'T': 16, 'V': 17, 'W': 18, 'Y': 19}


def train(lr, epochs, batch_size):
    x, y = get_data()
    X = preprocess_data(x)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # Create an instance of the model
    model = MyModel()

    # define loss function and optimizer
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels, sample_weight):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions, sample_weight=sample_weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels, sample_weight):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions, sample_weight=sample_weight)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    epochs_list, test_loss_list, train_loss_list = iterate(test_accuracy, test_ds, test_loss, test_step, train_accuracy,
                                                           train_ds, train_loss, train_step, epochs)
    # plot loss as a function of number of epochs
    plot_loss(epochs_list, train_loss_list, test_loss_list)

    # print confusion matrix
    y_pred = model.predict(x_test)
    y_pred2 = np.argmax(y_pred, axis=1)
    y_test2 = np.argmax(y_test, axis=1)
    matrix = confusion_matrix(y_test2, y_pred2)
    print(matrix)

    # plot the ROC curve
    plot_roc(model, x_test, y_test2)

    # calc recall
    rc = tf.keras.metrics.Recall()
    rc.update_state(y_test2, y_pred2)
    cur_recall = rc.result().numpy()

    # calc precision
    pr = tf.keras.metrics.Precision()
    pr.update_state(y_test2, y_pred2)
    cur_precision = pr.result().numpy()

    # find best peptides from new data
    find_covid_peptid(model)
    return epochs_list, test_loss_list, cur_recall, cur_precision


"""
iterations for learning the weights of the model 
"""


def iterate(test_accuracy, test_ds, test_loss, test_step, train_accuracy, train_ds, train_loss, train_step, epochs):
    test_loss_list = []
    train_loss_list = []
    epochs_list = []
    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for data, labels in train_ds:
            sample_weight = np.argmax(labels, axis=1) * 2 + 1
            train_step(data, labels, sample_weight)

        for test_images, test_labels in test_ds:
            sample_weight = np.argmax(test_labels, axis=1) * 2 + 1
            test_step(test_images, test_labels, sample_weight)
        test_loss_list.append(test_loss.result())
        train_loss_list.append(train_loss.result())
        epochs_list.append(epoch)
        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
    return epochs_list, test_loss_list, train_loss_list


"""
find 5 most probable positive peptids of spike protein 
"""


def find_covid_peptid(model):
    new_data = np.genfromtxt('dasas', dtype='str')
    process_new_data = preprocess_data(new_data)
    new_data_preds = model.predict(process_new_data)
    pos_preds = np.asarray([lst[1] for lst in new_data_preds])
    conc = np.vstack([range(pos_preds.shape[0]), pos_preds]).T
    sorted_conc = conc[conc[:, 1].argsort()][::-1]
    for i in range(PEPTIDS_TO_RETURN):
        print(new_data[int(sorted_conc[i, 0])] + " prob: " + str(sorted_conc[i, 1]))





# DATA PREPROCESSING FUNCTIONS

"""
get data and labels from files
"""


def get_data():
    x_pos = np.genfromtxt('./pos_A0201.txt', dtype='str')
    y_pos = np.ones((x_pos.shape[0]))
    x_neg = np.genfromtxt('./neg_A0201.txt', dtype='str')
    x_neg = np.random.choice(x_neg, 10000)
    y_neg = np.zeros((x_neg.shape[0]))
    x = np.concatenate([x_pos, x_neg])
    y = np.concatenate([y_pos, y_neg])
    y = to_categorical(y)
    return x, y


"""
preprocess data into one hot representation
"""


def preprocess_data(data):
    new_data = []
    words_count = 0
    for word in data:
        word_mat = np.zeros(((9,)))
        index = 0
        for letter in word:
            ltr_index = amino_acids_dict[letter]
            word_mat[index] = ltr_index
            index += 1
        new_data.append(word_mat)
        words_count += 1
    new_data = np.asarray(new_data)
    new_data = to_categorical(new_data)
    return new_data




# VISUALIZATION FUNCTIONS

"""
 plot train and test loss as function of epoch
"""


def plot_loss(epochs, train_loss, test_loss):
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, test_loss, label='test')
    plt.title('Train and Test loss as function of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()


"""
plot roc curve for the model
"""


def plot_roc(model, x_test, y_test):
    y_pred_keras = [lst[1] for lst in model.predict(x_test)]
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras, pos_label=True)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


"""
plot loss as function of epochs per lr value
"""


def plot_loss_lr(epochs, loss):
    plt.plot(epochs, loss[0], 'g', label='lr = 0.1')
    plt.plot(epochs, loss[1], 'b', label='lr = 0.01')
    plt.plot(epochs, loss[2], 'r', label='lr = 0.001')
    plt.plot(epochs, loss[3], 'm', label='lr = 0.0001')

    plt.title('Test loss as function of epochs for each learning rate value')
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.show()


"""
plot loss as function of epochs per batch size value
"""


def plot_loss_batch(epochs, loss):
    plt.plot(epochs, loss[0], 'g', label='bs = 1')
    plt.plot(epochs, loss[1], 'b', label='bs = 16')
    plt.plot(epochs, loss[2], 'r', label='bs = 64')
    plt.plot(epochs, loss[3], 'm', label='bs = 256')

    plt.title('Test loss as function of epochs for each batch size value')
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # HYPER - PARAMETERS
    EPOCHS = 50
    BATCH_SIZE = 128
    LR = 0.001

    epoch_list, test_loss_list, recall, precision = train(LR, EPOCHS, BATCH_SIZE)
    print(recall)
    print(precision)

    # # run for all batch sizes
    # bs_list = [1,16,64,256]
    # loss_list = []
    # epoch_list = []
    # recall_list, precision_list, f1_list  = [], [], []
    # for bs in bs_list:
    #     epoch_list, test_loss_list, recall, precision = train(LR, EPOCHS, bs)
    #     loss_list.append(test_loss_list)
    #     recall_list.append(recall)
    #     precision_list.append(precision)
    #     f1_list.append(2*precision*recall/(precision+recall))
    # plot_loss_batch(epoch_list, loss_list)
    # print(recall_list)
    # print(precision_list)
    # print(f1_list)
    #
    # # run for all lr values
    # lr_list = [0.1,0.01,0.001,0.0001]
    # loss_list = []
    # epoch_list = []
    # recall_list, precision_list, f1_list  = [], [], []
    # for lr in lr_list:
    #     epoch_list, test_loss_list, recall, precision = train(lr, EPOCHS, BATCH_SIZE)
    #     loss_list.append(test_loss_list)
    #     recall_list.append(recall)
    #     precision_list.append(precision)
    #     f1_list.append(2*precision*recall/(precision+recall))
    # plot_loss_lr(epoch_list, loss_list)
    # print(recall_list)
    # print(precision_list)
    # print(f1_list)
