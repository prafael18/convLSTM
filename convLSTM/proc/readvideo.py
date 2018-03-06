import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

#Image constants
IMAGE_WIDTH = 135
IMAGE_HEIGHT = 240
BASE_DIR = "/home/rafael/Documents/unicamp/ic/src/data"
TRAIN_DIR = "train"
VAL_DIR = 'val'
TEST_DIR = "test"
# Global constants describing the CIFAR-10 data set.
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def inputs(train=True):

    if train:
        data_dir = os.path.join(BASE_DIR, VAL_DIR)
    else:
        data_dir = os.path.join(BASE_DIR, TEST_DIR)

    print(data_dir)
    input_dir = os.path.join(data_dir, "inputs")
    label_dir = os.path.join(data_dir, "labels")
    inputs = os.listdir(input_dir)
    labels = os.listdir(label_dir)
    for i, f in enumerate(inputs):
        if i == 0:
            file_path = os.path.join(input_dir, f)
            print(file_path)
            cap = cv2.VideoCapture(file_path)
            print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while(True):
                ret, frame = cap.read()
                if ret:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    print(lab.shape)
                    sm_lab_frame = cv2.resize(lab, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_CUBIC)
                    print(sm_lab_frame)
                    sm_frame = cv2.resize(frame, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_CUBIC)
                    plt.subplot(211)
                    plt.imshow(sm_frame)
                    plt.subplot(212)
                    plt.imshow(sm_lab_frame)
                    plt.show()
                    while True:
                        pass
                    # cv2.imshow('inter_video', sm_gray_frame)
                    # if cv2.waitKey(25) & 0xFF == ord('q'):
                    #     break
                    # print(sm_frame.shape)
                    # print(sm_frame)
                else:
                    break
        else:
            break
    return

if __name__ == "__main__":
    inputs()