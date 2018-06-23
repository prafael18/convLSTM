#!/usr/bin/python3

"""Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format with the provided 3 color channels.
 Allows to subsequently train a neural network in TensorFlow with the generated tfrecords.
 Due to common hardware/GPU RAM limitations, this implementation allows to limit the number of frames per
 video actually stored in the tfrecords. The code automatically chooses the frame step size such that there is
 an equal separation distribution of the video images. Implementation supports Optical Flow
 (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel.
"""


from tensorflow.python.platform import gfile
from matplotlib import pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import optparse
import os
import matplotlib.pyplot as plt

STD_SCORE = 1
FT_SCALE = 0
VIDEO_NORM = 2
FRAME_NORM = 1
RAW = 0
RGB = 1
LAB = 0


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
      if total_record_num > 1:
        filename = os.path.join(destination_path, name + "_" + str(current_record_num+1) + "_" + str(clip_count+1) + ".tfrecords")
      else:
        filename = os.path.join(destination_path, name + ".tfrecords")
      print('Writing', filename)
      writer = tf.python_io.TFRecordWriter(filename)

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
  n_array = n_array.astype(np.float32)
  init_shape = n_array.shape
  n_array = n_array.reshape(np.product(n_array.shape[:-1]), n_array.shape[-1])
  if type == STD_SCORE:
    n_array -= np.mean(n_array, axis=0)
    n_array /= np.std(n_array, axis=0)
  elif type == FT_SCALE:
    min = np.min(n_array, axis=0)
    max = np.max(n_array, axis=0)
    if max.all() > 0:
      n_array = (n_array-min)/(max-min)
  n_array = n_array.reshape(init_shape)
  return n_array


def plot_frame(prev_image, new_image, title):
  prev_image = prev_image.astype(np.uint8)
  new_image = new_image.astype(np.uint8)
  plt.title(title)
  plt.subplot(211)
  plt.imshow(prev_image)
  plt.subplot(212)
  plt.imshow(new_image)
  plt.show()


def convert_video_to_numpy(record_num, filenames, width, height, n_channels, max_frames_per_clip, input, norm_dim, norm_type,
                           colorspace, label_dtype, clip_list=None):
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
    video = []
    clip = np.array([])


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
        if input:
          if colorspace:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        else:
          if label_dtype == "float32":
              frame = normalize(frame, FT_SCALE)
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_CUBIC)

        if np.sum(image) == 0:
          ret, frame = getNextFrame(cap)
          continue

        if input and norm_dim == FRAME_NORM:
          image = normalize(image, norm_type)
        #if not input:
          #image = image.reshape(image.shape[0], image.shape[1], 1)
          # image = normalize(image, FT_SCALE)
          #image = image.reshape(image.shape[:-1])

        if not clip.any():
          clip = image.reshape(1, height, width, n_channels)
        else:
          clip = np.concatenate((clip, image.reshape(1, height, width, n_channels)), axis=0)

        frames_counter += 1

        if frames_counter == max_frames_per_clip:
          if input and norm_dim == VIDEO_NORM:
            clip = normalize(clip, norm_type)
          video.append(clip)

          num_clips += 1
          frames_counter = 0
          clip = np.array([])

        if input:
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
                               max_frames_per_video, clips_per_file, norm_dim, norm_type, colorspace):
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

  input_filenames = []
  for ip in input_path:
      for fn in gfile.Glob(os.path.join(ip, file_suffix)):
          input_filenames.append(fn)

  label_filenames = []
  for lp in label_path:
      for fn in gfile.Glob(os.path.join(lp, file_suffix)):
          label_filenames.append(fn)

  print(input_filenames)
  print(label_filenames)


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
    label_data, clip_list = convert_video_to_numpy(record_num=i, filenames=label_filenames_split[i],
                                                    width=width, height=height, n_channels=label_channels, max_frames_per_clip=max_frames_per_video,
                                                    input=False, norm_dim=norm_dim, norm_type=norm_type, colorspace=colorspace, label_dtype=label_dtype)

    #When input=True, an empty frame_list is returned.
    input_data, _ = convert_video_to_numpy(record_num=i, filenames=input_filenames_split[i],
                                           width=width, height=height, n_channels=input_channels, max_frames_per_clip=max_frames_per_video,
                                           input=True, norm_dim=norm_dim, norm_type=norm_type, colorspace=colorspace, label_dtype=label_dtype, clip_list=clip_list)

    for j in range(label_data.shape[0]):
      print("Record {} - Video {}: label clips = {}, input clips = {}".
            format(i+1, j+1, label_data[j].shape[0], input_data[j].shape[0]))

    save_numpy_to_tfrecords(input_data, label_data, destination_path,
                            dataset_name, i, total_record_number, input_dtype, label_dtype,
                            max_frames_per_video, clips_per_file)


if __name__ == "__main__":
  parser = optparse.OptionParser()

  parser.add_option("-m", "--machine",
                    action="store", type="int", dest="machine",
                    help="Machine index where execution takes place. 0 - local, 1 - neuron0, 2 - neuron1.",
                    default=0)
  parser.add_option("-s", "--dataset",
                    action="store", type="string", dest="dataset",
                    help="Dataset that should be processed. Either test, train or val.",
                    default="val")
  parser.add_option("-c", "--color_space",
                    action="store", type="string", dest="color_space",
                    help="Color space to which we should convert input data (by default labels will be converted to grayscale)."
                         "Accepts rgb and lab.",
                    default="rgb")
  parser.add_option("-t", "--norm_type",
                    action="store", type="string", dest="norm_type",
                    help="Normalization type for input data. May choose between standard score (ss) and feature scale (fs).",
                    default=None)
  parser.add_option("-n", "--norm_dim",
                    action="store", type="string", dest="norm_dim",
                    help="Normalization dimension for input data. May choose between clip norm (vnorm), frame norm (fnorm) and raw.",
                    default="raw")
  parser.add_option("-l", "--label_norm",
                    action="store", type="string", dest="label_norm",
                    help="Normalization type for label data. May choose between raw and feature scale (fs).",
                    default="raw")
  parser.add_option("-f", "--num_files",
                    action="store", type="string", dest="num_files",
                    help="Should tfrecords be saved into a single file or 1 file per clip (many).",
                    default="single")
  parser.add_option("-d", "--dest_dir",
                    action="store", type="string", dest="dest_dir",
                    help="Destination dir that should be appended to machine destination in case of creating many files.",
                    default="out")

  options, args = parser.parse_args()

# Dynamically set variables

  if options.label_norm == "raw":
      label_dtype = "uint8"
  elif options.label_norm == "fs":
      label_dtype = "float32"
  else:
      exit(1)

  if options.norm_type == "ss":
    norm_type = STD_SCORE
  elif options.norm_type == "fs":
    norm_type = FT_SCALE
  else:
    norm_type = None

  if options.norm_dim == "vnorm":
    norm_dim = VIDEO_NORM
  elif options.norm_dim == "fnorm":
    norm_dim = FRAME_NORM
  elif options.norm_dim == "raw":
    norm_dim = RAW
  else:
    exit(1)

  if options.norm_dim == "raw":
    input_dtype = "uint8"
  else:
    input_dtype = "float32"

  if options.color_space == "rgb":
    colorspace = RGB
  elif options.color_space == "lab":
    colorspace = LAB
  else:
    exit(1)

  if options.num_files == "many":
    clips_per_file = 1
    videos_per_record = 1
  elif options.num_files == "single":
    clips_per_file = 1000
    videos_per_record = 40
  else:
    exit(1)

  print("norm_type is {}".format(options.norm_type))
  print("norm_dim is {} ".format(options.norm_dim))

  if not options.norm_type:
    tfrecord_name = "{}_{}_{}".format(options.dataset, options.norm_dim, options.color_space)
  else:
    tfrecord_name = "{}_{}_{}_{}".format(options.dataset, options.norm_type, options.norm_dim, options.color_space)

  print("tfrecord_name is {}".format(tfrecord_name))

  if options.machine == 0:
    destination = "/home/rafael/Documents/unicamp/ic/src/data/" + options.dataset + "/tfr"
    input_source_dir = ["/home/rafael/Documents/unicamp/ic/src/data/" + options.dataset + "/inputs"]
    label_source_dir = ["/home/rafael/Documents/unicamp/ic/src/data/" + options.dataset + "/labels"]
  elif options.machine == 1:
    destination = "/home/panda/ic/data/" + options.dataset
    input_source_dir = ["/home/panda/raw_data/" + options.dataset + "/inputs"]
    label_source_dir = ["/home/panda/raw_data/" + options.dataset + "/labels"]
    if options.dataset == "train":
        input_source_dir.append("/home/panda/augm_data/inputs")
        label_source_dir.append("/home/panda/augm_data/labels")
  elif options.machine == 2:
    destination = "/home/rafael/Documents/unicamp/ic/src/convLSTM/proc/test/tfr"
    input_source_dir = ["/home/rafael/Documents/unicamp/ic/src/convLSTM/proc/test/input"]
    label_source_dir = ["/home/rafael/Documents/unicamp/ic/src/convLSTM/proc/test/label"]
  elif options.machine == 3:
    destination = "/home/rafael/Documents/ic/src/data/" + options.dataset + "/tfr"
    input_source_dir = ["/home/rafael/Documents/ic/src/data/" + options.dataset + "/inputs"]
    label_source_dir = ["/home/rafael/Documents/ic/src/data/" + options.dataset + "/labels"]
    if options.dataset == "train":
        input_source_dir.append("/home/rafael/Documents/ic/src/data/augm_out/inputs")
        label_source_dir.append("/home/rafael/Documents/ic/src/data/augm_out/labels")
  else:
    exit(1)

  if options.dest_dir != "out":
    destination += ("/" + options.dest_dir)
    if not os.path.isdir(destination):
      os.mkdir(destination)


# Static variables
  width_video = 240
  height_video = 135
  input_channels = 3
  label_channels = 1
  max_frames_per_video = 15
  # label_dtype = "float32"
  file_suffix = "*.avi"

  print("""Options selected:
        machine = {}
        colorspace = {}
        norm_type = {}
        norm_dim = {}
        num_files = {}
        destination = {}
        input_source = {}
        label_source = {}
        tfrecord_filename = {}""".format(options.machine, colorspace, norm_type, norm_dim, options.num_files,
                                      destination, input_source_dir, label_source_dir, tfrecord_name))

  convert_videos_to_tfrecord(input_source_dir, label_source_dir, destination, tfrecord_name,
                             videos_per_record, file_suffix,
                             width_video, height_video, input_channels, label_channels,
                             input_dtype, label_dtype,
                             max_frames_per_video,
                             clips_per_file, norm_dim, norm_type, colorspace)
