import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from scipy.misc import imread, imshow
import cv2
import os

# tfrecord_path = '/home/rafael/Documents/unicamp/ic/src/data/scp/tfr/*'
# save_path = '/home/rafael/Documents/unicamp/ic/src/data/scp/videos'

# tfrecord_path = '/home/rafael/Documents/ic/src/convLSTM/proc/test/*'
# save_path = '/home/rafael/Documents/ic/src/convLSTM/proc/test'

# tfrecord_path = '/home/rafael/Documents/ic/src/data/train/tfr/raw_rgb_lfs/*.tfrecords'
# tfrecord_path = '/home/rafael/Documents/ic/src/data/train/tfr/*'
tfrecord_path = '/home/rafael/Documents/unicamp/ic/src/data/dhf1k/train/train_2_*'
save_path = '/home/rafael/Documents/ic/src/'

height = None
width = None

# def feed
# def read_and_decode():

def pixel_intensity(input_video):
    print("Whole clip max is {} and min is {}".format(np.max(input_video), np.min(input_video)))
    for n in range(input_video.shape[3]):
        print("Whole clip channel {} max is {} and min is {}.".format(n+1, np.max(input_video[:,:,:,n]), np.min(input_video[:,:,:,n])))
        for m in range(input_video.shape[0]):
            print("Frame {} channel {} max is {} and min is {}.".format(m + 1, n + 1, np.max(input_video[m][:,:,n]),
                                                                       np.min(input_video[m][:,:,n])))
    print("\n\n\n")

def stats(input_video):
    print(input_video.shape)
    for channel in range(input_video.shape[3]):
        print("For the entire clip channel {}:\nnp.std()={}, np.mean()={}"
              .format(channel+1, np.std(input_video[:,:,:,channel]), np.mean(input_video[:,:,:,channel])))
    for frame in range(input_video.shape[0]):
        print("For entire frame {}:\nnp.std()={}, np.mean()={}"
              .format(frame+1, np.std(input_video[frame]), np.mean(input_video[frame])))
        for channel in range(input_video.shape[3]):
            print("For frame {} channel {}:\nnp.std()={}, np.mean={}"
                  .format(frame+1, channel+1, np.std(input_video[frame][:, :, channel]),
                          np.mean(input_video[frame][ :, :, channel])))


def plot_video(title, input_video, label_video, height, width):
  plt.subplot(211)
  plt.title(title)
  plt.imshow(input_video[0])
  plt.subplot(212)
  plt.title("Label")
  plt.imshow(label_video[0].reshape(height, width), cmap="Greys_r")
  plt.show()


def save_video(name, input_video):
  print("gonna save here")
  print(input_video.shape)
  input_video = map(input_video, 0, 255)
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(os.path.join(save_path, name + ".avi"), fourcc, 3, (240, 135))
  for frame in input_video:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
  out.release()

def map(input_video, output_start, output_end):
  slope = (output_end - output_start) / (np.max(input_video)-np.min(input_video))
  output = output_start + slope * (input_video - np.min(input_video))
  return np.uint8(output)

def readTFRecord(filepaths):

    files = 0
    for path in filepaths:
      record_iterator = tf.python_io.tf_record_iterator(path=path)
      filename = path.split('/')[-1]
      print(filename)

      cnt = 0
      for i, string_record in enumerate(record_iterator):
          cnt += 1
      print("Number of clips in record = ", cnt)
      files += 1
    print("Total files = ", files)
    exit(1)


    for path in filepaths:
      record_iterator = tf.python_io.tf_record_iterator(path=path)
      filename = path.split('/')[-1]
      print(filename)

      for i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature['height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['width']
                    .int64_list
                    .value[0])
        # video_id = int(example.features.feature['video_id']
        #             .int64_list
        #             .value[0])
        # clip_id = int(example.features.feature['clip_id']
        #             .int64_list
        #             .value[0])

        frames = int(example.features.feature['num_frames']
                     .int64_list
                     .value[0])
        print("Number of frames in this record is ", frames)

        if "ss" in filename or "fs" in filename:
          dtype = np.float32
        else:
          dtype = np.uint8

        print(filename)
        input_byte_list = example.features.feature['input'].bytes_list.value
        input_string_list = [np.fromstring(input_string, dtype=dtype) for input_string in input_byte_list]
        input_video = np.reshape(input_string_list, (frames, height, width, -1))

        imshow(input_video[0])

        label_byte_list = example.features.feature['label'].bytes_list.value
        label_string_list = [np.fromstring(label_string, dtype=np.uint8) for label_string in label_byte_list]
        label_video = np.reshape(label_string_list, (frames, height, width))

        imshow(label_video[0])

        if i == 0:
            # pixel_intensity(input_video)
            # stats(input_video)
            pixel_intensity(label_video)
            stats(label_video)

        if "ss" in filename:
          input_video = (input_video - np.min(input_video))/(np.max(input_video) - np.min(input_video))

        label_video = np.reshape(label_video, label_video.shape[:-1])
        print(label_video.shape)
        print(video_id, clip_id)
        if clip_id == 0 and video_id == 29:
            plt.imshow(label_video[0])
            plt.show()
        # print("Video shape = ", input_video.shape)

        if "raw" in filename:
          video_name = '_'.join(filename.split('_')[:3])
        else:
          video_name = '_'.join(filename.split('_')[:4])

        # print (video_name)

        # save_video(video_name, input_video)

if __name__ == "__main__":
  filenames = gfile.Glob(tfrecord_path)
  # print(filenames)
  readTFRecord(filenames)
