import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings

import input
import config
import model

tfrecords_filename = config.eval["tfrecords_filename"]
load_model_dir = config.eval["load_model_dir"]
num_epochs = config.eval["num_epochs"]
batch_size = config.eval["batch_size"]
image_height = config.eval["image_height"]
image_width = config.eval["image_width"]
input_channels = config.eval["input_channels"]
label_channels = config.eval["label_channels"]


def eval(epoch = None):
    warnings.simplefilter("error", RuntimeWarning)
    with tf.Session() as sess:

        train_dataset = tf.data.TFRecordDataset(tfrecords_filename)
        train_dataset = train_dataset.shuffle(buffer_size=40)

        train_dataset = train_dataset.map(input.parse_function)
        train_dataset = train_dataset.repeat(1)
        train_dataset = train_dataset.batch(batch_size=batch_size)

        # Runs through tfrecord once. Must call initializer for every epoch
        iterator = train_dataset.make_one_shot_iterator()

        input_batch, label_batch = iterator.get_next()

        #Loads SavedModel, initalizing graph and variables
        model.load(sess, load_model_dir, tags=[tf.saved_model.tag_constants.TRAINING])

        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

        g = tf.get_default_graph()

        #Tensors that will be fed when running op.
        input_tensor = g.get_tensor_by_name("input_x:0")
        label_tensor = g.get_tensor_by_name("label_y:0")

        # Necessary in order for input and output to have well defined shapes

        logits = g.get_tensor_by_name("logits:0")
        activations = tf.nn.sigmoid(logits, name="pred_activations")
        loss = g.get_tensor_by_name("loss:0")
        # iterator_train = g.get_operation_by_name("MakeIterator")
        # sess.run(iterator.initializer)
        sim_list = []
        cc_list = []
        mse_list = []
        loss_list = []
        while True:
            try:
                input_x, label_y = sess.run([input_batch, label_batch])
                feed_dict = {
                    input_tensor: input_x,
                    label_tensor: label_y
                }
                pred, mean_loss = sess.run([activations, loss], feed_dict = feed_dict)

                # print(input_x.shape)
                # print(label_y.shape)
                # print(pred[0].shape)

                pred = np.reshape(pred[0], pred[0].shape[:3])
                label = np.reshape(label_y, label_y.shape[1:4])
                # input_x = np.reshape(input_x, input_x.shape[1:])

                # plt.subplot(211)
                # plt.plot("Prediction")
                # plt.imshow(pred[200])
                # plt.subplot(212)
                # plt.title("Label")
                # plt.imshow(label[200])
                # plt.show()
                # print("Average pixel value for:\nPred = ", np.mean(pred))
                # print("Label = ", np.mean(label))
                loss_list.append(mean_loss)
                sim_list.append(sim(pred.copy(), label.copy()))
                cc_list.append(cc(pred.copy(), label.copy()))
                mse_list.append(mse(pred.copy(), label.copy()))

                # print(pred.shape)
            except tf.errors.OutOfRangeError:
                print("Finished testing predictions")

                sim_value = np.mean(np.array(sim_list))
                cc_value = np.mean(np.array(cc_list))
                mse_value = np.mean(np.array(mse_list))
                loss_value = np.mean(np.array(loss_list))

                if epoch:
                    print("Epoch {}'s loss in validation set: {}".format(epoch, loss_value))
                print("SIM = {}\nCC = {}\nMSE = {}\n".format(sim_value, cc_value, mse_value))

                return sim_value, cc_value, mse_value

def sim(pred, label):
    warnings.simplefilter("error", RuntimeWarning)
    #Pre-process data:
    #(1) Normalize pred and label between 0 and 1
    #(2) Make sure that all pixel values add up to 1
    # print("Sum pred = ", np.sum(pred))

    assert pred.shape[0] is label.shape[0]
    frames = pred.shape[0]

    sim_list = []

    for i in range(frames):
        try:
            pred[i] = (pred[i] - np.min(pred[i]))/(np.max(pred[i])-np.min(pred[i]))
            pred[i] = pred[i]/np.sum(pred[i])
            label[i] = label[i]/np.sum(label[i])
            sim_coeff = np.minimum(pred, label)
            sim_list = [np.sum(f) for f in sim_coeff]
        except RuntimeWarning:
            pass
            # print("Failed to append frame {} to sim_list".format(i))

    return np.mean(np.array(sim_list))


def cc(pred, label):
    warnings.simplefilter("error", RuntimeWarning)
    frames = pred.shape[0]
    assert frames is label.shape[0]

    corr_coeff = []

    for i in range(frames):
        try:
            # Normalize data to have mean 0 and variance 1
            pred[i] = (pred[i] - np.mean(pred[i])) / np.std(pred[i])
            label[i] = (label[i] - np.mean(label[i])) / np.std(label[i])

            # Calculate correlation coefficient for every frame
            pd = pred[i] - np.mean(pred[i])
            ld = label[i] - np.mean(label[i])
            corr_coeff.append((pd * ld).sum() / np.sqrt((pd * pd).sum() * (ld * ld).sum()))
        except RuntimeWarning:
            pass
            # print("Failed to append frame {} to corr_coeff".format(i))

    return np.mean(np.array(corr_coeff))

def mse(pred, label):

    mean_squared_error = []
    for i in range(label.shape[0]):
        mean_squared_error.append(np.mean((pred[i]-label[i])**2))

    return np.mean(np.array(mean_squared_error))


if __name__ == '__main__':
    eval()