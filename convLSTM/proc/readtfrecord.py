import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tfrecords_filename = '/home/rafael/Documents/unicamp/ic/src/data/val/tfr/val_fs_raw_rgb_1_1.tfrecords'

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

def readTFRecord(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    count = 0
    for i, string_record in enumerate(record_iterator):
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

        if i == 0:
            pixel_intensity(input_video)
            #
            # plt.subplot(211)
            # plt.title("Input")
            # plt.imshow(input_video[0])
            # plt.subplot(212)
            # plt.title("Label")
            # plt.imshow(label_video[0].reshape(height, width), cmap="Greys_r")
            # plt.show()

        for j in range(input_video.shape[0]):
            if np.sum(input_video[j]) == 0:
                print("Found frame {} input clip {} that is black".format(j, i))
        for j in range(label_video.shape[0]):
            if np.sum(label_video[j]) == 0:
                print("Found frame {} in label clip {} that is black".format(j, i))
                print(np.sum(label_video[j]))
                plt.imshow(input_video[j].reshape((135, 240, 3)))
                plt.show()

if __name__ == "__main__":
    readTFRecord(tfrecords_filename)