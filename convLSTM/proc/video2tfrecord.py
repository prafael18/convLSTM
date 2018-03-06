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

FLAGS = flags.FLAGS

#Directory specific flags
flags.DEFINE_integer('n_videos_in_record', 40, 'Number of videos stored in one single tfrecord file. If there are too many videos inside the directory,'
                                               'reducing the number of videos per record might aid with any memory problems.')
flags.DEFINE_string('file_suffix', "*.avi", 'defines the video file type, e.g. .mp4')


#Data specific flags
flags.DEFINE_integer('width_video', 240, 'the desired width of the videos to be stored in tfrecord')
flags.DEFINE_integer('height_video', 135, 'the desired height of the videos to be stored in tfrecord')
flags.DEFINE_integer('input_channels', 3, 'the input channels source video (e.g. 3 for RGB)')
flags.DEFINE_integer('label_channels', 1, 'the output channels for label video (e.g. 1 for grayscale)')
flags.DEFINE_string('input_dtype', "uint8", 'data type to use for numpy matrices')
flags.DEFINE_string('label_dtype', "float32", 'data type to use for numpy matrices')



#Flags that require adjustments
flags.DEFINE_string('destination', '/home/rafael/Documents/unicamp/ic/src/data/test/tfr', 'Directory for storing tf records')
flags.DEFINE_string('input_source', '/home/rafael/Documents/unicamp/ic/src/data/test/inputs', 'Directory with input video files')
flags.DEFINE_string('label_source', '/home/rafael/Documents/unicamp/ic/src/data/test/labels', 'Directory with label video files')
flags.DEFINE_string('dataset_name', 'test', 'name used to create tfrecord file')

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
  if ret == False:
    return np.array([])

  return np.asarray(frame)

def save_numpy_to_tfrecords(input_data, label_data, destination_path, name,
                            current_record_num, total_record_num, input_dtype, label_dtype):
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
  num_videos = input_data.shape[0]
  height = input_data[0].shape[1]
  width = input_data[0].shape[2]
  writer = None
  feature = {}

  for videoCount in range(num_videos):

    if videoCount == 0:
      if writer is not None:
        writer.close()
      filename = os.path.join(destination_path,
                              name + str(current_record_num+1) + '_of_' + str(total_record_num) + '.tfrecords')

      print('Writing', filename)
      writer = tf.python_io.TFRecordWriter(filename)

    num_frames = input_data[videoCount].shape[0]

    input_list = []
    label_list = []
    for imageCount in range(num_frames):

      input_list.append(input_data[videoCount][imageCount, :, :, :]
                        .astype(input_dtype)
                        .tostring())

      label_list.append(label_data[videoCount][imageCount, :, :, :]
                        .astype(label_dtype)
                        .tostring())


    feature['input'] = _bytes_feature(input_list)
    feature['label'] = _bytes_feature(label_list)
    feature['height'] = _int64_feature(height)
    feature['width'] = _int64_feature(width)
    feature['num_frames'] = _int64_feature(num_frames)

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

  print("Finished writing {} videos to tfrecord file {}/{}".format(num_videos, current_record_num+1, total_record_num))

  if writer is not None:
    writer.close()


def convert_video_to_numpy(record_num, filenames, width, height, n_channels, input, frame_list=None):
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

  data = []
  frames = []

  def video_file_to_ndarray(i, filename):

    assert os.path.isfile(filename), "Couldn't find video file"
    cap = getVideoCapture(filename)
    assert cap is not None, "Couldn't load video capture:" + filename + ". Moving to next video."

    frames_counter = 0

    frame = getNextFrame(cap)
    video = np.array([])

    while frame.any():
      max_frames = -1
      if frame_list is not None:
        max_frames = frame_list[i]

      #Change image colorspace (n_channels is 3 for input and 1 for label)
      if input:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
      else:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = (image-np.min(image))/(np.max(image)-np.min(image))

      #Reduce image dimensions with bicubic interpolation.
      image = cv2.resize(image, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_CUBIC)

      if not video.any():
        video = image.reshape(1, height, width, n_channels)
      else:
        video = np.concatenate((video, image.reshape(1, height, width, n_channels)), axis=0)

      frames_counter += 1
      if frames_counter == max_frames:
        break

      frame = getNextFrame(cap)

    v = video.copy()
    cap.release()
    return v, frames_counter

  print("Generating numpy arrays from video data:")
  for i, file in enumerate(filenames):
    try:
      v, f = video_file_to_ndarray(i, file)
      print("{} of {} videos within {} record {}: {}".format(i+1, number_of_videos,
                "input" if input else "label",
                record_num+1, filenames[i]))
      data.append(v)
      if not input:
        frames.append(f)
    except Exception as e:
      print(e)

  return np.array(data), frames



def convert_videos_to_tfrecord(input_path, label_path, destination_path, dataset_name,
                               n_videos_in_record, file_suffix,
                               width, height, input_channels, label_channels, input_dtype, label_dtype):
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
    label_data, frame_list = convert_video_to_numpy(record_num = i, filenames=label_filenames_split[i],
                  width=width, height=height, n_channels=label_channels,
                  input=False)

    #When input=True, an empty frame_list is returned.
    input_data, _ = convert_video_to_numpy(record_num = i, filenames=input_filenames_split[i],
                  width=width, height=height, n_channels=input_channels,
                  input=True, frame_list=frame_list)

    for j in range(label_data.shape[0]):
      print("Record {} - Video {}: label frames = {}, input frames = {}".
            format(i+1, j+1, label_data[j].shape[0], input_data[j].shape[0]))

    save_numpy_to_tfrecords(input_data, label_data, destination_path,
                            dataset_name, i, total_record_number, input_dtype, label_dtype)

def main(argv):
  convert_videos_to_tfrecord(FLAGS.input_source, FLAGS.label_source, FLAGS.destination, FLAGS.dataset_name,
                             FLAGS.n_videos_in_record, FLAGS.file_suffix,
                             FLAGS.width_video, FLAGS.height_video, FLAGS.input_channels, FLAGS.label_channels,
                             FLAGS.input_dtype, FLAGS.label_dtype)


if __name__ == '__main__':
  app.run()
