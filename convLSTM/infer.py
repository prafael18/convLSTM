import tensorflow as tf
import os
import optparse
import numpy as np
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt
import model


FLOAT32 = 1

verbose = False
dtype = 0


def parse_function(serialized_example):
    """Processes an example ProtoBuf and returns input and label with shape:
        [batch_size, n_frames, height, width, channels]
    """
    # Parse into tensors
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'num_frames': tf.FixedLenFeature([], tf.int64),
            'input': tf.VarLenFeature(tf.string),
            'label': tf.VarLenFeature(tf.string)
        }
    )

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    num_frames = tf.cast(features['num_frames'], tf.int32)

    dense_input = tf.sparse_tensor_to_dense(features['input'], default_value='*')
    dense_label = tf.sparse_tensor_to_dense(features['label'], default_value='*')

    if dtype == FLOAT32:
        in_dtype = tf.float32
    else:
        in_dtype = tf.uint8

    input_list = tf.decode_raw(dense_input, in_dtype)
    label_list = tf.decode_raw(dense_label, tf.float32)

    # Height and width are dynamically calculated.
    # Therefore, tf.reshape() will cause shape to be undefined and throw an
    # error when running inference for model.
    input_shape = tf.stack([num_frames, height, width, 3])
    label_shape = tf.stack([num_frames, height, width, 1])

    inputs = tf.reshape(input_list, input_shape)
    labels = tf.reshape(label_list, label_shape)

    inputs = tf.cast(inputs, tf.float32)
    labels = tf.cast(labels, tf.float32)

    return inputs, labels


def sim(sim_list, pred, label):
    # warnings.simplefilter("error", RuntimeWarning)

    # Pre-process data:
    # (1) Normalize pred and label between 0 and 1
    # (2) Make sure that all pixel values add up to 1

    num_videos = pred.shape[0]
    num_frames = pred.shape[1]
    frame_sim_list = []

    for v in range(num_videos):
        for f in range(num_frames):
            pred[v][f] = (pred[v][f] - np.min(pred[v][f]))/(np.max(pred[v][f])-np.min(pred[v][f]))
            pred[v][f] = pred[v][f]/np.sum(pred[v][f])
            label[v][f] = label[v][f]/np.sum(label[v][f])
            min_val = np.minimum(pred[v][f], label[v][f])
            sim_coeff = np.sum(min_val)
            frame_sim_list.append(sim_coeff)
        sim_list.append(np.mean(np.array(frame_sim_list)))


def cc(cc_list, pred, label):
    # Pred and label have shapes (batch_size, frames, height, width, channels)
    # warnings.simplefilter("error", RuntimeWarning)

    num_videos = pred.shape[0]
    num_frames = pred.shape[1]

    corr_coeff = []
    for v in range(num_videos):
        for f in range(num_frames):
            # Normalize data to have mean 0 and variance 1
            pred[v][f] = (pred[v][f] - np.mean(pred[v][f])) / np.std(pred[v][f])
            label[v][f] = (label[v][f] - np.mean(label[v][f])) / np.std(label[v][f])

            # Calculate correlation coefficient for every frame
            pd = pred[v][f] - np.mean(pred[v][f])
            ld = label[v][f] - np.mean(label[v][f])
            corr_coeff.append((pd * ld).sum() / np.sqrt((pd * pd).sum() * (ld * ld).sum()))
        cc_list.append(np.mean(np.array(corr_coeff)))
    return


def mse(mse_list, pred, label):

    num_videos = pred.shape[0]
    num_frames = pred.shape[1]

    mean_squared_error = []

    for v in range(num_videos):
        for f in range(num_frames):
           mean_squared_error.append(np.mean((pred[v][f]-label[v][f])**2))
    mse_list.append(np.mean(np.array(mean_squared_error)))


def eval(tf_filename, load_model_dir, batch_size):

    with tf.Session() as sess:

        # Get test dataset to make predictions on.
        test_dataset = tf.data.TFRecordDataset(tf_filename)
        test_dataset = test_dataset.map(parse_function)
        # test_dataset = test_dataset.repeat(1)
        test_dataset = test_dataset.batch(batch_size=batch_size)

        # Runs through tfrecord once. Must call initializer for every epoch
        iterator = test_dataset.make_one_shot_iterator()

        input_batch, label_batch = iterator.get_next()

        # Loads SavedModel, initalizing graph and variables
        model.load(sess, load_model_dir, tags=[tf.saved_model.tag_constants.TRAINING])

        g = tf.get_default_graph()

        # Tensors that will be fed when running op.
        input_tensor = g.get_tensor_by_name("input_x:0")
        label_tensor = g.get_tensor_by_name("label_y:0")

        # Retrieve prediction tensors
        logits = g.get_tensor_by_name("logits:0")
        sig_logits = tf.nn.sigmoid(logits)
        # activations = sig_logits
        activations = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_tensor)
        loss = g.get_tensor_by_name("loss:0")

        # Retrieve epoch counter
        epoch_counter = [v for v in tf.global_variables() if v.name == "epoch_counter:0"][0]
        # print("Current epoch is {}".format(epoch_counter.eval()))

        sim_list = []
        cc_list = []
        mse_list = []
        loss_list = []
        i = 0
        while True:
            try:
                input_x, label_y = sess.run([input_batch, label_batch])
                feed_dict = {
                    input_tensor: input_x,
                    label_tensor: label_y
                }

                pred, mean_loss, logits_out, sig_logits_out = sess.run([activations, loss, logits, sig_logits], feed_dict=feed_dict)

                print("Max of pred batch is {}".format(np.max(pred)))
                print("Min of pred batch is {}".format(np.min(pred)))
                loss_list.append(mean_loss)
                sim(sim_list, pred.copy(), label_y.copy())
                cc(cc_list, pred.copy(), label_y.copy())
                mse(mse_list, pred.copy(), label_y.copy())

                i += 1
                if i == 1:
                    plt.title("label")
                    plt.hist(label_y[0][0].flatten())
                    plt.show()
                    plt.title("predictions")
                    plt.hist(pred[0][0].flatten())
                    plt.show()
                    plt.title("logits")
                    plt.hist(logits_out[0][0].flatten())
                    plt.show()
                    plt.title("sigmoid(logits)")
                    plt.hist(sig_logits_out[0][0].flatten())
                    plt.show()

            except tf.errors.OutOfRangeError:

                sim_value = np.mean(np.array(sim_list))
                cc_value = np.mean(np.array(cc_list))
                mse_value = np.mean(np.array(mse_list))
                loss_value = np.mean(np.array(loss_list))

                print("Average loss is {:.4f}\n"
                      "Average similarity (SIM) is {:.4f}\n"
                      "Average pearson's correlation coefficient (CC) is {:.4f}\n"
                      "Average mean squared error (MSE) is {:.4f}\n"
                      .format(loss_value, sim_value, cc_value, mse_value))

                return sim_value, cc_value, mse_value


if __name__ == '__main__':
    base_model_dir = "/home/rafael/Documents/ic/src/results/exp"
    base_tfrecord_dir = "/home/rafael/Documents/ic/src/data/test/tfr/*"
    batch_size = 5

    # Parse options
    parser = optparse.OptionParser()
    parser.add_option("-e", "--experiment",
                      action="store", type="int", dest="experiment",
                      help="Experiment number which we wish to make predictions for.",
                      default=14)
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose",
                      help="Print helpful data.", default=False)
    options, args = parser.parse_args()

    # Get cmdline args
    verbose = options.verbose
    exp = options.experiment

    # Retrieve directory from which to load model
    load_model_dir = os.path.join(base_model_dir, str(exp))

    # Retrieve tfrecords filename
    tf_filename = None

    filepaths = gfile.Glob(base_tfrecord_dir)
    readme_file = os.path.join(load_model_dir, "README.md")
    with open(readme_file, "r") as f:
        name = f.readline().strip()
        keys = name.split("_")

    # Find tfrecord among all tfrecords using README keys
    for fp in filepaths:
        tf_filename = fp
        for k in keys:
            if k not in fp:
                tf_filename = None
                break
        if tf_filename:
            break

    if "ss" in tf_filename or "fs" in tf_filename:
        dtype = FLOAT32

    eval(tf_filename, load_model_dir, batch_size)
