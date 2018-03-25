import tensorflow as tf
import numpy as np
import shutil
import os
import cv2


# input has shape [batch_size, time_steps, img_size_x, img_size_y, channels]

LEARNING_RATE = 0.001

def save(sess, save_dir, overwrite=False, **builder_kwargs):
    """
    Saves metagraph and weights of current session into directory save_dir.
    """
    if not "tags" in builder_kwargs:
        builder_kwargs["tags"] = []
    if os.path.isdir(save_dir):
        if overwrite:
            shutil.rmtree(save_dir)
        else:
            raise Exception("'{}' exists".format(save_dir))
    builder = tf.saved_model.builder.SavedModelBuilder(save_dir)
    builder.add_meta_graph_and_variables(sess, **builder_kwargs)
    builder.save()

def load(sess, save_dir, **builder_kwargs):
    """
    Loads metagraph and weights to current session from data in save_dir.
    """
    if not "tags" in builder_kwargs:
        builder_kwargs["tags"] = []
    tf.saved_model.loader.load(sess, export_dir=save_dir, **builder_kwargs)

def max_pool(input, **kwargs):
    with tf.name_scope("max_pool"):
        output = tf.layers.max_pooling2d(input, **kwargs)
    return output

def upconv(input, **kwargs):
    with tf.name_scope("upconv"):
        output = tf.layers.conv2d_transpose(input, **kwargs)
    return output

def conv(input, **kwargs):
    with tf.name_scope("conv"):
        output = tf.layers.conv2d(input, **kwargs)
        print("Conv output op_name = " , output)
    return output

def resize(input, **kwargs):
    with tf.name_scope("resize"):
        output = tf.image.resize_image_with_crop_or_pad(input, **kwargs)
        print("Resize output shape = ", output)
    return output

def convLSTM(input, **kwargs):
    with tf.name_scope("convLSTM") as scope:
        print("Current scope is: ", scope)
        print ("ConvLSTM input_shape: ", input.shape)
        print("Batch_size: ", tf.shape(input)[0])
        # # input_shape = input.shape.as_list()[2:]
        input_shape = [34, 60, kwargs["output_channels"]]
        lstmCell = tf.contrib.rnn.Conv2DLSTMCell(
            input_shape=input_shape, **kwargs)

        # print(input.shape.as_list())
        # print(tf.shape(input[0]))

        initial_state = lstmCell.zero_state(tf.shape(input)[0],
            dtype=tf.float32)
        outputs, state= tf.nn.dynamic_rnn(lstmCell, input,
            initial_state=initial_state,
            dtype=tf.float32,
            scope=scope)
    return outputs

def framewise_op(input, op, **kwargs):
    print("Conv input_shape: ", input.shape)

    #Returns a list of tensors [-1, height, width, channels]
    #Note that this only works because each of these variables are well defined
    #in the operation set_shape inside the input module.
    input_shape = tf.concat([[-1], tf.shape(input)[2:]], axis=0)

    input_flat = tf.reshape(input, input_shape)

    output_flat = op(input_flat, **kwargs)

    output_shape = tf.concat([tf.shape(input)[:2], tf.shape(output_flat)[1:]], axis=0)
    output = tf.reshape(output_flat, output_shape)

    print("Conv output_shape: ", output.shape)
    # output.set_shape([None, None, 135, 240, kwargs['filters']])

    return output

def inference(inputs, name=None):
    """Build model up to where it can be used for inference"""

    #All 5x5 kernels (including input-to-state) with 128->64->64 hidden states
    # batch_size = tf.placeholder(tf.int32, [None], name='batch_size')

    net = framewise_op(inputs, conv,
            filters=32,
            kernel_size=[5, 5],
            padding="SAME", activation=tf.nn.relu)
    net = framewise_op(net, max_pool,
            pool_size=[2,2],
            strides=[2,2],
            padding="SAME")
    net = framewise_op(net, conv,
            filters=64,
            kernel_size=[5,5],
            padding="SAME", activation=tf.nn.relu)
    net = framewise_op(net, max_pool,
            pool_size=[2,2],
            strides=[2,2],
            padding="SAME")
    print("After all conv and max_pool ops:", net)
    net = convLSTM(net,
            output_channels=128,
            kernel_shape=[5, 5],
            initializers=tf.contrib.layers.xavier_initializer(),
            forget_bias=1.0)
    net = convLSTM(net,
            output_channels=64,
            kernel_shape=[5,5],
            initializers=tf.contrib.layers.xavier_initializer(),
            forget_bias=1.0)
    net = convLSTM(net,
            output_channels=64,
            kernel_shape=[5,5],
            initializers=tf.contrib.layers.xavier_initializer(),
            forget_bias=1.0)
    net = framewise_op(net, upconv,
            filters=64,
            kernel_size=[5,5],
            strides=[2,2],
            activation=tf.nn.relu,
            padding="SAME")
    net = framewise_op(net, upconv,
            filters=32,
            kernel_size=[5,5],
            strides=[2,2],
            activation=tf.nn.relu,
            padding="SAME")
    net = framewise_op(net, conv,
            filters=1,
            kernel_size=[1, 1],
            padding="SAME")
    output = framewise_op(net, resize,
            target_height=135,
            target_width=240)

    print("Logits op name before rename = ", output)

    output = tf.identity(output, name=name)

    print("Logits op name = ", output)
    return output

def loss(logits, labels, name=None):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
            labels=labels,
            name="cross_entropy_batch")
    print("Cross entropy = ", cross_entropy)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name=name)
    tf.summary.scalar("cross_entropy", cross_entropy_mean)
    return cross_entropy_mean

def train(loss, global_step, name=None):
    print(loss)
    tf.summary.scalar("loss", loss)
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss, global_step=global_step, name=name)
    return train_op
