import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tfrecords_filename = '/home/rafael/Documents/unicamp/ic/src/data/train/tfr/train1_of_1.tfrecords'

# def feed
# def read_and_decode():


def readTFRecord(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    count = 0
    for i, string_record in enumerate(record_iterator):
        count += 1
        # if i < 10:
        #     print("Reading record ", i)
        #     example = tf.train.Example()
        #     example.ParseFromString(string_record)
        #     height = int(example.features.feature['height']
        #                  .int64_list
        #                  .value[0])
        #
        #     width = int(example.features.feature['width']
        #                 .int64_list
        #                 .value[0])
        #     frames = int(example.features.feature['num_frames']
        #                  .int64_list
        #                  .value[0])
        #     # print("Number of frames in this record is ", frames)
        #     input_byte_list = example.features.feature['input'].bytes_list.value
        #     input_string_list = [np.fromstring(input_string, dtype=np.uint8) for input_string in input_byte_list]
        #     input_video = np.reshape(input_string_list, (frames, height, width, -1))
        #     # print("Video shape = ", input_video.shape)
        #
        #     label_byte_list = example.features.feature['label'].bytes_list.value
        #     label_string_list = [np.fromstring(label_string, dtype=np.float32) for label_string in label_byte_list]
        #     label_video = np.reshape(label_string_list, (frames, height, width, -1))
        #
        #
        #     # print("For whole video, max = {} and min = {}".format(np.max(label_video), np.min(label_video)))
        #     print("Label shape = {}, Input shape = {}".format(label_video.shape, input_video.shape))
        #
        #     plt.subplot(211)
        #     plt.title("Input")
        #     plt.imshow(input_video[0])
        #     plt.subplot(212)
        #     plt.title("Label")
        #     plt.imshow(label_video[0].reshape(height, width), cmap="Greys_r")
        #     plt.show()
    print (count)

if __name__ == "__main__":
    readTFRecord(tfrecords_filename)