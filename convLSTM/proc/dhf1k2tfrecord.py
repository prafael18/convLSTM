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
    'clips_per_file': [5, 40],
    'frames_per_clip': [18, 18],
    'samples_per_video': [1, 1], # Make sure that clips_per_file%samples_per_video==0 to avoid creating unnecessary files
    'base_data_dir': ['/home/rafael/Documents/unicamp/ic/data/dhf1k', '/home/panda/raw_data/dhf1k'],
    'base_out_dir': ['/home/rafael/Documents/unicamp/ic/src/data/dhf1k', '/home/panda/ic/data/dhf1k']
}

def cleanDir(tfr_base_dir):
    dirs = [os.path.join(tfr_base_dir, 'train'), os.path.join(tfr_base_dir, 'val')]
    for d in dirs:
        d = os.path.join(d, '*')
        fps = glob.glob(d)
        for f in fps:
            os.remove(f)

# Accepts a 4D numpy array with nr of frames on axis 0.
# Returns 5D array with samples_per_video clips randomly sampled from uniform bins
def getRandomSamples(input_np, label_np, samples_per_video, frames_per_clip):
    # Ensure we have enough frames to sample from
    assert samples_per_video*frames_per_clip < input_np.shape[0]

    input_list, label_list = ([], [])
    bin = input_np.shape[0]//samples_per_video
    for i in range(samples_per_video):
        # Determine starting point within bin to extract clip
        s = np.random.randint(i*bin, (i+1)*bin-frames_per_clip)
        # Extract samples from video
        input_sample = input_np[s:s+frames_per_clip]
        label_sample = label_np[s:s+frames_per_clip]

        # Add sample clips to list with one extra dim
        input_list.append(np.reshape(input_sample, (1,)+input_sample.shape))
        label_list.append(np.reshape(label_sample, (1,)+label_sample.shape))

    return np.concatenate(input_list, axis=0), np.concatenate(label_list, axis=0)

# Both input and label are 5D arrays of shape (clips, frames, height, width, channels)
def numpy2record(input_np, label_np, out_dir, id, clips_per_file, frames_per_clip):
    # Sanity check
    assert input_np.shape[0] == label_np.shape[0]

    # Get data params
    height = input_np.shape[2]
    width = input_np.shape[3]

    # Dataset name can be either train or val
    dataset_name = os.path.basename(out_dir)

    nr_clips = input_np.shape[0]
    nr_files = math.ceil(nr_clips/clips_per_file)

    print('Writing video %s:' % str(id).zfill(4))
    cur_clip = 0
    for i in range(nr_files):
        tfr_basename = '%s_%s_%d_of_%d.tfrecords'%(dataset_name, str(id).zfill(3), i+1, nr_files)
        tfr_filename = os.path.join(out_dir, tfr_basename)
        writer = tf.python_io.TFRecordWriter(tfr_filename)

        clip_range = clips_per_file if (nr_clips - cur_clip) >= clips_per_file else (nr_clips - cur_clip)
        for j in range(clip_range):
            # Calculate offset
            # file_offset = i*clips_per_file*frames_per_clip
            # clip_offset = j*frames_per_clip
            # offset = file_offset + clip_offset

            input_clip = input_np[j]
            label_clip = label_np[j]

            input_list = [input_clip[i].astype(np.uint8).tostring() for i in range(frames_per_clip)]
            label_list = [label_clip[i].astype(np.uint8).tostring() for i in range(frames_per_clip)]

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

def makeTFrecords(input_dir, label_dir, out_dir, samples_per_video, clips_per_file, frames_per_clip):
    input_fps = glob.glob(os.path.join(input_dir, '*'))
    input_fps.sort()
    # Compute total number of clips to record
    total_clips = samples_per_video*(len(input_fps))
    # Initialize lists of clips
    input_list, label_list = ([],[])
    # Keep track of clip number in order to record last few clips when we dont have enough clips to fill a file
    clip_nr = 0
    for ifp in input_fps:
        # Get label filepath
        id = int(os.path.basename(ifp).split('.')[0])
        str_id = str(id).zfill(4)
        label_maps_dir = os.path.join(label_dir, str_id, 'maps')

        print("Processing video %d/%d..."%(id, len(input_fps)))

        # Convert video to numpy ndarray
        input_np, label_np = video2numpy(ifp, label_maps_dir)

        # Get 5D array of shape (clips, frames, height, width, channels)
        input_np, label_np = getRandomSamples(input_np, label_np, samples_per_video, frames_per_clip)

        # Add array to clip list
        input_list.append(input_np)
        label_list.append(label_np)

        # Update number of clips recorded
        clip_nr += samples_per_video

        if (len(input_list) >= clips_per_file or clip_nr == total_clips):
            input_np = np.concatenate(input_list, axis=0)
            label_np = np.concatenate(label_list, axis=0)

            # Convert numpy ndarray to tfrecord
            numpy2record(input_np, label_np, out_dir, id, clips_per_file, frames_per_clip)

            #Reset data lists
            input_list, label_list = ([],[])

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-m', '--machine', action='store', type='int', dest='machine',
                        help='Machine in which execution takes place.\n\t0 -> local\t1 -> neuron0',
                        default=0)
    parser.add_option('-r', '--rm', action='store_true', dest='remove',
                        help='Whether or not to clean tfrecord directory',
                        default=False)
    options, args = parser.parse_args()
    m = options.machine
    rm = options.remove

    clips_per_file = PARAMS['clips_per_file'][m]
    samples_per_video = PARAMS['samples_per_video'][m]
    frames_per_clip = PARAMS['frames_per_clip'][m]
    base_data_dir = PARAMS['base_data_dir'][m]
    base_out_dir = PARAMS['base_out_dir'][m]

    if rm:
        cleanDir(base_out_dir)

    train_input_dir = os.path.join(base_data_dir, 'train', 'inputs')
    train_label_dir = os.path.join(base_data_dir, 'train', 'labels')
    train_tfr_dir = os.path.join(base_out_dir, 'train')

    makeTFrecords(train_input_dir, train_label_dir, train_tfr_dir, samples_per_video, clips_per_file, frames_per_clip)

    val_input_dir = os.path.join(base_data_dir, 'val', 'inputs')
    val_label_dir = os.path.join(base_data_dir, 'val', 'labels')
    val_tfr_dir = os.path.join(base_out_dir, 'val')


    makeTFrecords(val_input_dir, val_label_dir, val_tfr_dir, samples_per_video, clips_per_file, frames_per_clip)
