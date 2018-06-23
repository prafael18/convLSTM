#!/usr/bin/python3
import tensorflow as tf
import optparse
import numpy as np
import cv2
from tensorflow.python.platform import gfile
import warnings
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import model


INPUT_LABEL = 1
INPUT_PRED = 2
PRED_LABEL = 3
FLOAT32 = 1

dtype = None
verbose = False


def sim(list, pred, label):
    # Pre-process data:
    # (1) Normalize pred and label between 0 and 1
    # (2) Make sure that all pixel values add up to 1
    # print("Sum pred = ", np.sum(pred))

    num_videos = pred.shape[0]
    num_frames = pred.shape[1]

    sim_list = []

    for v in range(num_videos):
        for f in range(num_frames):
            pred[v][f] = (pred[v][f] - np.min(pred[v][f]))/(np.max(pred[v][f])-np.min(pred[v][f]))
            pred[v][f] = pred[v][f]/np.sum(pred[v][f])
            label[v][f] = label[v][f]/np.sum(label[v][f])
            sim_coeff = np.minimum(pred[v][f], label[v][f])
            sim_list.append(np.sum(sim_coeff))
    list.append(np.mean(np.array(sim_list)))
    return


def cc(cc_list, pred, label):
    # Pred and label have shapes (batch_size, frames, height, width, channels)
    warnings.simplefilter("error", RuntimeWarning)

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
    label_list = tf.decode_raw(dense_label, tf.uint8)

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


def save_video(name, save_dir, upper_vid, upper_vid_conv, lower_vid, lower_vid_conv):
    # Input videos are batches of 5 clips.
    # Videos are 5-d tensors of shape (batch_size, frames, height, width, channels)

    # Get input properties
    clips = upper_vid.shape[0]
    frames = upper_vid.shape[1]
    height = upper_vid.shape[2]
    width = upper_vid.shape[3]

    # Create filename
    save_file = os.path.join(save_dir, name+".avi")
    if verbose:
        print("Saving video to file: {}".format(save_file))

    # Get VideoWriter object. We multiply by 2x the height because the clips are concatenated vertically
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_file, fourcc, 5, (width, 2*height))

    for v in range(clips):
        for f in range(frames):
            upper_frame = map(upper_vid[v][f], 0, 255)
            lower_frame = map(lower_vid[v][f], 0, 255)
            upper_frame = cv2.cvtColor(upper_frame, upper_vid_conv)
            lower_frame = cv2.cvtColor(lower_frame, lower_vid_conv)
            frame = np.concatenate((upper_frame, lower_frame), axis=0)
            out.write(frame)
    out.release()


def map(input_video, output_start, output_end):
    slope = (output_end - output_start) / (np.max(input_video)-np.min(input_video))
    output = output_start + slope * (input_video - np.min(input_video))
    return np.uint8(output)


def eval(tf_filename, load_model_dir, name, mode, batch_size):

    with tf.Session() as sess:

        # Get test dataset to make predictions on.
        test_dataset = tf.data.TFRecordDataset(tf_filename)
        test_dataset = test_dataset.map(parse_function)
        test_dataset = test_dataset.repeat(1)
        test_dataset = test_dataset.batch(batch_size=batch_size)

        # Runs through tfrecord once. Must call initializer for every epoch
        iterator = test_dataset.make_one_shot_iterator()

        input_batch, label_batch = iterator.get_next()

        # Loads SavedModel, initalizing graph and variables
        model.load(sess, load_model_dir, tags=[tf.saved_model.tag_constants.TRAINING])

        g = tf.get_default_graph()

        # Get epoch tensor
        epoch_tensor = [v for v in tf.global_variables() if v.name == "epoch_counter:0"][0]

        # Tensors that will be fed when running op.
        input_tensor = g.get_tensor_by_name("input_x:0")
        label_tensor = g.get_tensor_by_name("label_y:0")

        # Retrieve prediction tensors
        logits = g.get_tensor_by_name("logits:0")
        activations = tf.nn.relu(logits, name="pred_activations")

        # Retrieve loss tensor
        loss = g.get_tensor_by_name("loss:0")

        batch_num = 0
        cc_list = []
        sim_list = []
        mse_list = []

        while True:
            try:
                input_x, label_y = sess.run([input_batch, label_batch])
                feed_dict = {
                    input_tensor: input_x,
                    label_tensor: label_y
                }

                pred, mean_loss = sess.run([activations, loss], feed_dict=feed_dict)

                cc(cc_list, pred.copy(), label_y.copy())
                sim(sim_list, pred.copy(), label_y.copy())
                mse(mse_list, pred.copy(), label_y.copy())

                batch_num += 1

                if batch_num == 20:
                    # Set upper video
                    if mode == INPUT_LABEL or mode == INPUT_PRED:
                        upper_vid = input_x
                        upper_vid_conv = cv2.COLOR_RGB2BGR
                    else:
                        upper_vid = pred
                        upper_vid_conv = cv2.COLOR_GRAY2BGR

                    # Set lower video
                    lower_vid_conv = cv2.COLOR_GRAY2BGR
                    if mode == INPUT_LABEL or mode == PRED_LABEL:
                        lower_vid = label_y
                    else:
                        lower_vid = pred

                    save_video(name, load_model_dir, upper_vid, upper_vid_conv, lower_vid, lower_vid_conv)

            except tf.errors.OutOfRangeError:
                epoch = epoch_tensor.eval()
                print("Epoch {}".format(epoch))
                print("CC = {:.4f}".format(np.mean(cc_list)))
                print("SIM = {:.4f}".format(np.mean(sim_list)))
                print("MSE = {:.4f}".format(np.mean(mse_list)))
                return

if __name__ == "__main__":
    base_model_dir = "/home/rafael/Documents/ic/src/results/exp"
    base_tfrecord_dir = "/home/rafael/Documents/ic/src/data/test/tfr/*.tfrecords"
    batch_size = 5

    # Parse options
    parser = optparse.OptionParser()
    parser.add_option("-m", "--mode",
                      action="store", type="string", dest="mode",
                      help="Consider i = input, l = label and p = predictions. Choose to generate "
                           "videos in any of the modes il, ip, pl.",
                      default="pl")
    parser.add_option("-e", "--experiment",
                      action="store", type="int", dest="experiment",
                      help="Experiment number which we wish to make predictions for.",
                      default=22)
    parser.add_option("-f", "--filename",
                      action="store", type="string", dest="filename",
                      help="Filename to name video after. Directory is by default the experiment's directory.",
                      default="pred_label")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose",
                      help="Print helpful data.", default=True)
    options, args = parser.parse_args()

    # Get cmdline args
    verbose = options.verbose
    mode = options.mode
    exp = options.experiment
    filename = options.filename

    # Set op mode
    if mode == "il": mode = INPUT_LABEL
    elif mode == "ip": mode = INPUT_PRED
    elif mode == "pl": mode = PRED_LABEL
    else: exit(1)

    # Retrieve directory from which to load model
    load_model_dir = os.path.join(base_model_dir, str(exp))

    # Retrieve tfrecords filename
    tf_filename = None

    filepaths = gfile.Glob(base_tfrecord_dir)
    readme_file = os.path.join(load_model_dir, "README.md")
    with open(readme_file, "r") as f:
        name = f.readline().strip()
        keys = name.split("_")

    # In case a filename is specified, we should use it instead.
    if filename is not None:
        name = filename

    # Find tfrecord among all tfrecords using README keys
    print(keys)
    print(filepaths)
    for fp in filepaths:
        tf_filename = fp
        for k in keys:
            if k not in fp:
                tf_filename = None
                break
        if tf_filename:
            break
    print(tf_filename)
    if "ss" in tf_filename or "fs" in tf_filename:
        dtype = FLOAT32

    if verbose:
        if mode == INPUT_LABEL: mode_str = "input-label"
        elif mode == INPUT_PRED: mode_str = "input-pred"
        else: mode_str = "pred-label"
        print("\nReading data from tfrecord: {}".format(tf_filename))
        print("Mode selected: {}".format(mode_str))
        print("Making videos for experiment {} processed as {}\n".format(exp, name))

    # Evaluate data and make predictions
    eval(tf_filename, load_model_dir, name, mode, batch_size)
