"""Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format with the provided 3 color channels.
 Allows to subsequently train a neural network in TensorFlow with the generated tfrecords.
 Due to common hardware/GPU RAM limitations, this implementation allows to limit the number of frames per
 video actually stored in the tfrecords. The code automatically chooses the frame step size such that there is
 an equal separation distribution of the video images. Implementation supports Optical Flow
 (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel.
"""

from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app

import cv2 as cv2
import numpy as np
import tensorflow as tf

import os
import readtfrecord

import matplotlib.pyplot as plt
from scipy.misc import imshow

FLAGS = flags.FLAGS

#Directory specific flags
flags.DEFINE_integer('n_videos_in_record', 40, 'Number of videos stored in one single tfrecord file. If there are too many videos inside the directory,'
                                               'reducing the number of videos per record might aid with any memory problems.')
flags.DEFINE_string('file_suffix', "*.avi", 'defines the video file type, e.g. .mp4')

STD_SCORE = 0
FEAT_SCALING = 1

#Data specific flags
flags.DEFINE_integer('width_video', 240, 'the desired width of the videos to be stored in tfrecord')
flags.DEFINE_integer('height_video', 135, 'the desired height of the videos to be stored in tfrecord')
flags.DEFINE_integer('input_channels', 3, 'the input channels source video (e.g. 3 for RGB)')
flags.DEFINE_integer('label_channels', 1, 'the output channels for label video (e.g. 1 for grayscale)')
flags.DEFINE_string('input_dtype', "float32", 'data type to use for numpy matrices')
flags.DEFINE_string('label_dtype', "float32", 'data type to use for numpy matrices')
flags.DEFINE_integer('max_frames_per_video', 15, 'maximum frames per clip')

set = "val"
#Flags that require adjustments
flags.DEFINE_integer('clips_per_file', 1000, 'Number of clips per tf record.')
flags.DEFINE_string('destination', '/home/rafael/Documents/unicamp/ic/src/data/'+set+'/tfr/', 'Directory for storing tf records')
flags.DEFINE_string('input_source', '/home/rafael/Documents/unicamp/ic/src/data/'+set+'/inputs/', 'Directory with input video files')
flags.DEFINE_string('label_source', '/home/rafael/Documents/unicamp/ic/src/data/'+set+'/labels/', 'Directory with label video files')
flags.DEFINE_string('dataset_name', set+"_fs_fnorm_rgb", 'name used to create tfrecord file')
flags.DEFINE_integer('norm', 1, 'Integer indicating normalization type (x - x.mean)/x.std:'
                                '0 - raw'
                                '1 - frame norm'
                                '2 - clip norm')

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def get_chunks(l, n):
  """Yield successive n-sized chunks from l.
  Used to create n sublists from a list l"""
  for i in range(0, len(l), n):
    yield l[i:i + n]


def getVideoCapture(path):
  assert os.path.isfile(path)
  cap = None
  if path:
    cap = cv2.VideoCapture(path)
  return cap


def getNextFrame(cap):
  ret, frame = cap.read()
  if not ret:
    return False, np.array([])
  return True, np.asarray(frame)

def save_numpy_to_tfrecords(input_data, label_data, destination_path, name,
                            current_record_num, total_record_num, input_dtype, label_dtype,
                            max_frames_per_clip, clips_per_file):
  """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
  :param input_data: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos, i=number of images, c=number of image
  channels, h=image height, w=image width
  :param label_data: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos, i=number of images, c=number of image
  channels, h=image height, w=image width
  :param destination_path: Directory to save the tfrecord files.
  :param name: filename; data samples type (train|valid|test)
  :param current_record_num: indicates the current record index (function call within loop)
  :param total_record_num: indicates the total number of record files.
  :param input_dtype: data type to use when saving pixel data to record.
  :param label_dtype: label data type to use when saving to tfrecord.
  """

  #Here we assume that the input and label videos have the same shape
  num_clips = input_data.shape[0]
  height = input_data[0].shape[1]
  width = input_data[0].shape[2]
  writer = None
  feature = {}

  print("Num videos = {}, height = {}, width = {}\n".format(num_clips, height, width))

  for clip_count in range(num_clips):
    if clip_count%clips_per_file == 0:
      if writer is not None:
        writer.close()

      filename = os.path.join(destination_path, name+".tfrecords")
      print('Writing', filename)
      writer = tf.python_io.TFRecordWriter(filename)
      #    name + "_" + str(current_record_num+1) + "_" + str(videoCount+1) + ".tfrecords")
      # '_of_' + str(total_record_num) + '.tfrecords')

    input_list = []
    label_list = []
    for imageCount in range(max_frames_per_clip):

      input_list.append(input_data[clip_count][imageCount, :, :, :]
                        .astype(input_dtype)
                        .tostring())

      label_list.append(label_data[clip_count][imageCount, :, :, :]
                        .astype(label_dtype)
                        .tostring())


    feature['input'] = _bytes_feature(input_list)
    feature['label'] = _bytes_feature(label_list)
    feature['height'] = _int64_feature(height)
    feature['width'] = _int64_feature(width)
    feature['num_frames'] = _int64_feature(max_frames_per_clip)

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

  print("Finished writing {} videos to tfrecord file {}/{}".format(num_clips, current_record_num+1, total_record_num))

  if writer is not None:
    writer.close()

def normalize(n_array, type):
  # print("Inside normalize")
  n_array = n_array.astype(np.float32)
  init_shape = n_array.shape
  n_array = n_array.reshape(np.product(n_array.shape[:-1]), n_array.shape[-1])
  # print(n_array.shape)
  if type == STD_SCORE:
    n_array -= np.mean(n_array, axis=0)
    n_array /= np.std(n_array, axis=0)
  if type == FEAT_SCALING:
    min = np.min(n_array, axis=0)
    max = np.max(n_array, axis=0)
    if max.all() > 0:
      n_array = (n_array-min)/(max-min)
  # print("Before reshape")
  # print("Channel 1: std = {}, mean = {}".format(np.std(n_array[:][i])))
  n_array = n_array.reshape(init_shape)
  # for i in range(n_array.shape[-1]):
  #   print("For channel {}: std = {}, mean = {}"
  #         .format(i+1, np.std(n_array[:][:][i]), np.mean(n_array[:][:][i])))
  return n_array


def convert_video_to_numpy(record_num, filenames, width, height, n_channels, max_frames_per_clip, input, norm, clip_list=None):
  """Generates an ndarray from multiple video files given by filenames.
  Implementation chooses frame step size automatically for a equal separation distribution of the video images.

  :param filenames
  :param width: desired frame width
  :param height: desired frame height
  :param n_channels: number of channels
  :param input: boolean indicating whether to process as input or label
  :param frame_list: list with number of frames to process in each video.
  :return
    data: Numpy array of shape (n_videos, frames, width, height, channels)
    frame_list: list of frames processed if input=False. Otherwise, returns empty list.
    """
  if not filenames:
    raise RuntimeError('No data files found.')

  number_of_videos = len(filenames)

  data = np.array([])
  clips = []

  print("List of clips per video: ", clip_list)

  def video_file_to_ndarray(i, filename):

    ignore_frames = []
    if filename.find('v35') >= 0:
      ignore_frames = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260]
    if filename.find('v12') >= 0:
      ignore_frames = [251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 400]

    assert os.path.isfile(filename), "Couldn't find video file"
    cap = getVideoCapture(filename)
    assert cap is not None, "Couldn't load video capture:" + filename + ". Moving to next video."

    total_frames_count = 0
    frames_counter = 0
    num_clips = 0

    ret, frame = getNextFrame(cap)
    # print(ret)
    # print(frame.shape)
    # print(frame.any())
    video = []
    clip = np.array([])

    # print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    empty_frames = []
    while ret:

      total_frames_count += 1

      if not frame.any():
        empty_frames.append(total_frames_count)

      if total_frames_count in ignore_frames:
        print("Ignoring frame that has frame.any() = ", frame.any())
        ret, frame = getNextFrame(cap)
        continue
      elif frame.any():
        # Change image colorspace (n_channels is 3 for input and 1 for label)
        frame = frame.astype(np.float32)
        if input:
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_CUBIC)

        if np.sum(image) == 0:
          ret, frame = getNextFrame(cap)
          continue

        if input and norm == 1:
          image = normalize(image, FEAT_SCALING)
        if not input:
          image = image.reshape(image.shape[0], image.shape[1], 1)
          image = normalize(image, FEAT_SCALING)
          image = image.reshape(image.shape[:-1])

        if not clip.any():
          clip = image.reshape(1, height, width, n_channels)
        else:
          clip = np.concatenate((clip, image.reshape(1, height, width, n_channels)), axis=0)

        frames_counter += 1

        if frames_counter == max_frames_per_clip:
          if input and norm == 2:
            clip = normalize(clip, FEAT_SCALING)
            # print("max = ", clip.max(), "min = ", clip.min())

          video.append(clip)

          # if input:
          #   readtfrecord.stats(clip)

          num_clips += 1
          frames_counter = 0
          clip = np.array([])

        if input:
          # print("Num clips = {}, clip_list[i] = {}".format(num_clips, clip_list[i]))
          if num_clips == clip_list[i]:
            break

        ret, frame = getNextFrame(cap)

      else:
        print("Breaking because frame.any() = ", frame.any())
        break

    cap.release()
    print("Empty frames = ", empty_frames)
    video = np.array(video)

    return video.copy(), num_clips

  print("Generating numpy arrays from video data:")
  for i, file in enumerate(filenames):
      v, n = video_file_to_ndarray(i, file)
      print("{} of {} videos (has {} clips) within {} record {}: {}".format(i+1, number_of_videos, n,
                "input" if input else "label",
                record_num+1, filenames[i]))
      if not data.any():
        data = v
      else:
        data = np.concatenate((data, v), axis=0)
      if not input:
        clips.append(n)

  print("Final record shape = ", data.shape)

  return data, clips



def convert_videos_to_tfrecord(input_path, label_path, destination_path, dataset_name,
                               n_videos_in_record, file_suffix,
                               width, height, input_channels, label_channels, input_dtype, label_dtype,
                               max_frames_per_video, clips_per_file, norm):
  """calls sub-functions convert_video_to_numpy and save_numpy_to_tfrecords in order to directly export tfrecords files
  :param input_path: directory where input videos videos are stored
  :param label_path: directory where label videos are stored
  :param destination_path: directory where tfrecords should be stored
  :param dataset_name: name to be given to tfrecord file
  :param n_videos_in_record: Number of videos stored in one single tfrecord file
  :param file_suffix: defines the video file type, e.g. *.mp4
  :param width: the width of the videos in pixels
  :param height: the height of the videos in pixels
  :param input_channels: specifies the number of channels the input videos have
  :param label_channels: specifies the number of channels label videos have
  :param dtype: Color depth as string for the images stored in the tfrecord files. Has to correspond to the source video color depth. '
                                                   'Specified as dtype (e.g. uint8 or uint16)
  :param max_frames_per_video: max_frames per video
  """

  input_filenames = gfile.Glob(os.path.join(input_path, file_suffix))
  label_filenames = gfile.Glob(os.path.join(label_path, file_suffix))


  if not input_filenames or not label_filenames:
    raise RuntimeError('Specified input or output directories have no files')

  print('Total input videos found: ' + str(len(input_filenames)))
  print('Total label videos found: ' + str(len(label_filenames)))


  if n_videos_in_record > len(input_filenames):
    total_record_number = 1
  else:
    total_record_number = int(np.ceil(len(input_filenames) / n_videos_in_record))

  input_filenames_split = list(get_chunks(input_filenames, n_videos_in_record))
  label_filenames_split = list(get_chunks(label_filenames, n_videos_in_record))

  print("Saving videos into {} record(s) where:".format(total_record_number))
  for k in range(input_filenames_split.__len__()):
    print("Record {} has {} videos.".format(k+1, input_filenames_split[k].__len__()))

  for i in range(total_record_number):

    #When calling convert_video_to_numpy with input=False, returns a list of the frames in the label data
    #which is used to ensure the number of frames processed in the label videos is the same as the input videos.
    label_data, clip_list = convert_video_to_numpy(record_num = i, filenames=label_filenames_split[i],
                                                    width=width, height=height, n_channels=label_channels, max_frames_per_clip=max_frames_per_video,
                                                    input=False, norm=norm)

    #When input=True, an empty frame_list is returned.
    input_data, _ = convert_video_to_numpy(record_num = i, filenames=input_filenames_split[i],
                                           width=width, height=height, n_channels=input_channels, max_frames_per_clip=max_frames_per_video,
                                           input=True, norm=norm, clip_list=clip_list)

    for j in range(label_data.shape[0]):
      print("Record {} - Video {}: label clips = {}, input clips = {}".
            format(i+1, j+1, label_data[j].shape[0], input_data[j].shape[0]))

    save_numpy_to_tfrecords(input_data, label_data, destination_path,
                            dataset_name, i, total_record_number, input_dtype, label_dtype,
                            max_frames_per_video, clips_per_file)

def main(argv):
  convert_videos_to_tfrecord(FLAGS.input_source, FLAGS.label_source, FLAGS.destination, FLAGS.dataset_name,
                             FLAGS.n_videos_in_record, FLAGS.file_suffix,
                             FLAGS.width_video, FLAGS.height_video, FLAGS.input_channels, FLAGS.label_channels,
                             FLAGS.input_dtype, FLAGS.label_dtype,
                             FLAGS.max_frames_per_video,
                             FLAGS.clips_per_file, FLAGS.norm)


if __name__ == '__main__':
  app.run()
