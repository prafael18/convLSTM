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
    warnings.simplefilter("error", RuntimeWarning)
    # Pre-process data:
    # (1) Normalize pred and label between 0 and 1
    # (2) Make sure that all pixel values add up to 1
    # print("Sum pred = ", np.sum(pred))

    num_videos = pred.shape[0]
    num_frames = pred.shape[1]

    sim_list = []

    for v in range(num_videos):
        for f in range(num_frames):
            try:
                pred[v][f] = (pred[v][f] - np.min(pred[v][f]))/(np.max(pred[v][f])-np.min(pred[v][f]))
                pred[v][f] = pred[v][f]/np.sum(pred[v][f])
                label[v][f] = label[v][f]/np.sum(label[v][f])
                sim_coeff = np.minimum(pred[v], label[v])
                sim_list = [np.sum(s) for s in sim_coeff]
            except RuntimeWarning:
                pass
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
            try:
                # Normalize data to have mean 0 and variance 1
                pred[v][f] = (pred[v][f] - np.mean(pred[v][f])) / np.std(pred[v][f])
                label[v][f] = (label[v][f] - np.mean(label[v][f])) / np.std(label[v][f])

                # Calculate correlation coefficient for every frame
                pd = pred[v][f] - np.mean(pred[v][f])
                ld = label[v][f] - np.mean(label[v][f])
                corr_coeff.append((pd * ld).sum() / np.sqrt((pd * pd).sum() * (ld * ld).sum()))
            except RuntimeWarning:
                pass
            # print("Failed to append frame {} to corr_coeff".format(i))
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

        # Tensors that will be fed when running op.
        input_tensor = g.get_tensor_by_name("input_x:0")
        label_tensor = g.get_tensor_by_name("label_y:0")

        # Retrieve prediction tensors
        logits = g.get_tensor_by_name("logits:0")
        # activations = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_tensor)
        activations = tf.nn.relu(logits, name="pred_activations")
        # activations = tf.losses.mean_squared_error(label_tensor, activations)

        loss = g.get_tensor_by_name("loss:0")

        input_x, label_y = sess.run([input_batch, label_batch])
        feed_dict = {
            input_tensor: input_x,
            label_tensor: label_y
        }

        pred, mean_loss = sess.run([activations, loss], feed_dict=feed_dict)

        print(pred.shape)
        print(input_x.shape)
        print(label_y.shape)


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


if __name__ == "__main__":
    base_model_dir = "/home/rafael/Documents/ic/src/results/exp"
    base_tfrecord_dir = "/home/rafael/Documents/ic/src/data/test/tfr/*"
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

    if verbose:
        if mode == INPUT_LABEL: mode_str = "input-label"
        elif mode == INPUT_PRED: mode_str = "input-pred"
        else: mode_str = "pred-label"
        print("\nReading data from tfrecord: {}".format(tf_filename))
        print("Mode selected: {}".format(mode_str))
        print("Making videos for experiment {} processed as {}\n".format(exp, name))

    # Evaluate data and make predictions
    eval(tf_filename, load_model_dir, name, mode, batch_size)

# print(pred.shape)
# print(pred[0].shape)
# pred = np.reshape(pred[0], pred[0].shape[:3])
# label = np.reshape(label_y, label_y.shape[1:4])
#     pred = cross_entropy_out
#
#     print(pred.shape)
#     print(label_y.shape)
#
#     loss_list.append(mean_loss)
#     sim(sim_list, pred.copy(), label_y.copy())
#     cc(cc_list, pred.copy(), label_y.copy())
#     mse(mse_list, pred.copy(), label_y.copy())
#
# except tf.errors.OutOfRangeError:
#     print("Finished testing predictions")
#
#     sim_value = np.mean(np.array(sim_list))
#     cc_value = np.mean(np.array(cc_list))
#     mse_value = np.mean(np.array(mse_list))
#     loss_value = np.mean(np.array(loss_list))
#
#     # if epoch:
#     #     print("Epoch {}'s loss in validation set: {}".format(epoch, loss_value))
#     print("SIM = {}\nCC = {}\nMSE = {}\n".format(sim_value, cc_value, mse_value))
#
#     return sim_value, cc_value, mse_value

# input_frame = np.reshape(input_x[0][3][:, :, :], input_x.shape[2:5])
# label_frame = np.reshape(label_y[0][3][:, :, :], label_y.shape[2:4])
# pred_frame = np.reshape(pred[0][3][:, :, :], pred.shape[2:4])
# ce_frame = np.reshape(cross_entropy_out[0][3][:, :, :], cross_entropy_out.shape[2:4])
#
# print(np.sum(np.abs(ce_frame-label_frame)))
# print(np.sum(np.abs(pred_frame-label_frame)))
#
# plt.subplot(211)
# plt.imshow(input_frame)
# plt.subplot(212)
# plt.imshow(label_frame)
# plt.show()
#
# print(label_frame.shape)
#
# diff = np.sum(np.abs(cross_entropy_out-pred))
# print(diff)
# print(label_y.shape)
# print(input_x.shape)
# print(cross_entropy_out.shape)
# print(cross_entropy_out)
# print(pred)
