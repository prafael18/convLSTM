import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tfrecords_filename = '/home/rafael/Documents/unicamp/ic/src/data/val/tfr/val1_of_1.tfrecords'

# def feed
# def read_and_decode():


def readTFRecord(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for i, string_record in enumerate(record_iterator):
        if i is not -1:
            print("Reading record ", i)
            example = tf.train.Example()
            example.ParseFromString(string_record)
            height = int(example.features.feature['height']
                         .int64_list
                         .value[0])

            width = int(example.features.feature['width']
                        .int64_list
                        .value[0])
            frames = int(example.features.feature['num_frames']
                         .int64_list
                         .value[0])
            # print("Number of frames in this record is ", frames)
            input_byte_list = example.features.feature['input'].bytes_list.value
            input_string_list = [np.fromstring(input_string, dtype=np.uint8) for input_string in input_byte_list]
            input_video = np.reshape(input_string_list, (frames, height, width, -1))
            # print("Video shape = ", input_video.shape)

            label_byte_list = example.features.feature['label'].bytes_list.value
            label_string_list = [np.fromstring(label_string, dtype=np.float32) for label_string in label_byte_list]
            label_video = np.reshape(label_string_list, (frames, height, width, -1))
            # print("Label video shape = ", label_video.shape)

            # input_string_2 = input_byte_list[1]
            # label_byte_list = example.features.feature['label'].bytes_list.value
            # print("Label byte list size = ", label_byte_list.size)
            # label_string = label_byte_list[frames]
            # input_1d = np.fromstring(input_string, dtype=np.uint8)
            # label_1d = np.fromstring(label_string, dtype=np.uint8)
            # input_video = np.array()
            # rec_input = np.reshape(input_1d, (height, width, -1))
            # rec_label = np.reshape(label_1d, (height, width))
            # print(rec_label[25:60, 50:100])
            # label_frame1 = np.array(input_video[0], dtype=np.float32)
            # label_frame1 = label_frame1/np.max(label_frame1)
            # slice = label_frame1.reshape(135,240)
            # print(slice.shape)
            # slice = slice[25:26]
            # print(slice)

            # label_frame1 = label_frame1.reshape(input_video.shape[1:3])
            # label_frame2 = input_video[6].reshape(input_video.shape[1:3])
            # max = np.max(label_frame1)
            # min = np.min(label_frame1)
            print("For whole video, max = {} and min = {}".format(np.max(label_video), np.min(label_video)))
            print(label_video.shape)
            plt.imshow(label_video[0].reshape(height, width), cmap="Greys_r")
            plt.show()
            # plt.subplot(211)
            # plt.imshow(, cmap="Greys_r")
            # plt.subplot(212)
            # plt.imshow(label_frame2, cmap="Greys_r")
            # plt.show()


if __name__ == "__main__":
    readTFRecord(tfrecords_filename)