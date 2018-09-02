# The basic idea is to have multiple tfrecord files per video, with a few clips in each.
# We have to process them one at a time in order to keep memory usage low.
# We use the video input filename to find the labels directory.
import tensorflow as tf
import numpy as np
import glob
import cv2
import os
import math
import optparse

from scipy.misc import imread, imshow

PARAMS = {
    'clips_per_file': [20, None],
    'frames_per_clip': [18, None],
    'base_data_dir': ['/home/rafael/Documents/unicamp/ic/data/dhf1k', None],
    'base_out_dir': ['/home/rafael/Documents/unicamp/ic/src/data/dhf1k', None]
}

def numpy2record(input_np, label_np, out_dir, id, clips_per_file, frames_per_clip):
    # Sanity check
    assert input_np.shape[0] == label_np.shape[0]

    # Get data params
    height = input_np.shape[1]
    width = input_np.shape[2]

    # Dataset name can be either train or val
    dataset_name = os.path.basename(out_dir)

    nr_frames = input_np.shape[0]
    nr_clips = nr_frames//frames_per_clip
    nr_files = math.ceil(nr_clips/clips_per_file)

    print('Processing video %s:' % str(id).zfill(4))
    cur_clip = 0
    for i in range(nr_files):
        tfr_basename = '%s_%s_%d_of_%d.tfrecords'%(dataset_name, str(id).zfill(3), i+1, nr_files)
        tfr_filename = os.path.join(out_dir, tfr_basename)
        writer = tf.python_io.TFRecordWriter(tfr_filename)

        clip_range = clips_per_file if (nr_clips - cur_clip) >= clips_per_file else (nr_clips - cur_clip)
        for j in range(clip_range):
            # Calculate offset
            file_offset = i*clips_per_file*frames_per_clip
            clip_offset = j*frames_per_clip
            offset = file_offset + clip_offset

            input_clip = input_np[offset:offset+frames_per_clip]
            label_clip = label_np[offset:offset+frames_per_clip]

            input_list = [input_np[i].astype(np.uint8).tostring() for i in range(frames_per_clip)]
            label_list = [label_np[i].astype(np.uint8).tostring() for i in range(frames_per_clip)]

            feature = {}
            feature['input'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=input_list))
            feature['label'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=label_list))
            feature['height'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
            feature['width'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))
            feature['num_frames'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[frames_per_clip]))

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            cur_clip += 1

        writer.close()
        print('\tFinished writing %d clips to tfrecord file %d/%d (%s)' % (clip_range, i+1, nr_files, tfr_filename))
    return


def video2numpy(input_fp, label_dir):
    # Retrieve video capture object
    cap = cv2.VideoCapture(input_fp)
    if not cap:
        print('Error: couldn\'t retrieve cap from file: %s' % input_fp)

    # Define list of numpy imgs
    input_np = []
    label_np = []

    ret, frame = cap.read()
    frame_nr = 1
    while ret:
        input_frame = cv2.resize(frame, dsize=(240, 135), interpolation=cv2.INTER_AREA)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        label_img = imread(os.path.join(label_dir, str(frame_nr).zfill(4)+'.png'))
        label_frame = cv2.resize(label_img, dsize=(240, 135), interpolation=cv2.INTER_AREA)
        label_frame = np.reshape(label_frame, label_frame.shape+(1,))

        # Append frames
        input_np.append(input_frame)
        label_np.append(label_frame)

        # Iterate while loop
        frame_nr += 1
        ret, frame = cap.read()

    input_np = np.stack(input_np, axis=0)
    label_np = np.stack(label_np, axis=0)

    return input_np, label_np

def makeTFrecords(input_dir, label_dir, out_dir, clips_per_file, frames_per_clip):
    input_fps = glob.glob(os.path.join(input_dir, '*'))
    for ifp in input_fps:
        # Get label filepath
        id = int(os.path.basename(ifp).split('.')[0])
        str_id = str(id).zfill(4)
        label_maps_dir = os.path.join(label_dir, str_id, 'maps')

        # Convert video to numpy ndarray
        input_np, label_np = video2numpy(ifp, label_maps_dir)

        # Convert numpy ndarray to tfrecord
        numpy2record(input_np, label_np, out_dir, id, clips_per_file, frames_per_clip)


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-m', '--machine', action='store', type='int', dest='machine',
                        help='Machine in which execution takes place.\n\t0 -> local\t1 -> neuron0',
                        default=0)
    options, args = parser.parse_args()
    m = options.machine

    clips_per_file = PARAMS['clips_per_file'][m]
    frames_per_clip = PARAMS['frames_per_clip'][m]
    base_data_dir = PARAMS['base_data_dir'][m]
    base_out_dir = PARAMS['base_out_dir'][m]

    train_input_dir = os.path.join(base_data_dir, 'train', 'inputs')
    train_label_dir = os.path.join(base_data_dir, 'train', 'labels')
    train_tfr_dir = os.path.join(base_out_dir, 'train')

    makeTFrecords(train_input_dir, train_label_dir, train_tfr_dir, clips_per_file, frames_per_clip)

    val_input_dir = os.path.join(base_data_dir, 'val', 'inputs', '*')
    val_label_dir = os.path.join(base_data_dir, 'val', 'labels', '*')
    val_tfr_dir = os.path.join(base_out_dir, 'val')

    makeTFrecords(val_input_dir, val_label_dir, val_tfr_dir, clips_per_file, frames_per_clip)
