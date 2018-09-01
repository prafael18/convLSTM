import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import time
import datetime
import optparse
import os

import model
import config
import infer

#i = config.train["machine_index"]
train_tfrecords_filename = None
val_tfrecords_filename = None
save_model_dir = None
load_model_dir = None
val_result_file = None
train_result_file = None
status_file = None
writer_dir = None
norm_type = None

num_epochs = config.train["num_epochs"]
batch_size = config.train["batch_size"]
image_height = config.train["image_height"]
image_width = config.train["image_width"]
input_channels = config.train["input_channels"]
label_channels = config.train["label_channels"]
val_epochs = config.train["val_epochs"]


def run_val(initializer, epoch_counter, logits, loss, feed_keys, feed_values, val_time):

    start_time = time.time()
    sess.run(initializer)
    sim_list = []
    cc_list = []
    mse_list = []
    loss_list = []

    # activations = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=feed_keys[1])
    # activations = tf.nn.sigmoid(logits)
    activations = tf.nn.relu(logits)

    while True:
        try:
            inputs, labels = sess.run([feed_values[0], feed_values[1]])
            feed_dict = {
                feed_keys[0]: inputs,
                feed_keys[1]: labels
            }

            pred, mean_loss = sess.run([activations, loss], feed_dict=feed_dict)

            loss_list.append(mean_loss)
            infer.sim(sim_list, pred.copy(), labels.copy())
            infer.cc(cc_list, pred.copy(), labels.copy())
            infer.mse(mse_list, pred.copy(), labels.copy())

        except tf.errors.OutOfRangeError:
            epoch = epoch_counter.eval()
            sim_value = np.mean(np.array(sim_list))
            cc_value = np.mean(np.array(cc_list))
            mse_value = np.mean(np.array(mse_list))
            loss_value = np.mean(np.array(loss_list))
            val_time.append(time.time()-start_time)

            print()
            print("Finished validating in {:.3f}s".format(val_time[-1]))
            print("Epoch {}'s loss in validation set: {:.4f}".format(epoch, loss_value))
            print("SIM = {:.3f}\nCC = {:.3f}\nMSE = {:.3f}".format(sim_value, cc_value, mse_value))
            print()

            with open(val_result_file, "a") as f:
                f.write("Epoch {}: LOSS = {:.3f} SIM = {:.3f} CC = {:.3f} MSE = {:.3f}\n"
                        .format(epoch, loss_value, sim_value, cc_value, mse_value))
            return mse_value


def run_train_epoch(initializer, epoch_counter, train_op, loss, l2_loss, feed_keys,
                    feed_values, train_time, summary_op, train_writer=None, run_options=None, run_metadata=None):

    run_metadata=None
    run_options=None

    sess.run(initializer)
    batch_loss = []
    batch_l2_loss = []
    start_time = time.time()
    count = 1
    while True:
        try:
            # print(feed_values[0])
            # print(feed_values[1])
            inputs, labels = sess.run([feed_values[0], feed_values[1]])
            feed_dict = {
                feed_keys[0]: inputs,
                feed_keys[1]: labels
            }
            # print("For epoch {} - batch {}: input shape = {} and label shape = {}".format(epoch_counter, count, inputs.shape, labels.shape))
            # for f in range(labels.shape[0]):
                # print(labels.shape)
                # print(labels[f][0,:,:,0].shape)
                # imsave("out/epoch_{}_batch_{}_input_{}.png".format(epoch+1, count, f), labels[f][0,:,:,0])
            # _, loss_val, summary = sess.run([train_op, loss, summary_op],
            #                                 feed_dict=feed_dict,
            #                                 options=run_options,
            #                                 run_metadata=run_metadata)
            _, loss_val, l2_loss_val = sess.run([train_op, loss, l2_loss],
                                            feed_dict=feed_dict)
            batch_loss.append(loss_val)
            batch_l2_loss.append(l2_loss_val)
            count+=1
        except tf.errors.OutOfRangeError:
            increment_epoch = epoch_counter.assign_add(1)
            sess.run([increment_epoch])
            epoch = epoch_counter.eval()
            epoch_time = time.time() -start_time
            with open(train_result_file, "a") as f:
                f.write("Epoch {}: TIME = {:.3f} LOSS = {:.3f} L2_LOSS = {:.3f}\n"
			.format(epoch, epoch_time, np.mean(np.array(batch_loss)), np.mean(batch_l2_loss)))
            print("Epoch {} completed in {} seconds.\nAverage cross-entropy loss is: {:.3}"
                  .format(epoch, epoch_time, np.mean(np.array(batch_loss))))
            train_time.append(epoch_time)
            # if epoch%1 == 0:
            #     train_writer.add_run_metadata(run_metadata, "Epoch {}".format(epoch))
            # train_writer.add_summary(summary, epoch)
            break
    return

def get_summary(load_model):
    if load_model:
        return tf.get_default_graph().get_tensor_by_name(name="Merge/MergeSummary:0")
    else:
        return tf.summary.merge_all()

def get_train_op(load_model, loss, global_step):
    if load_model:
        return tf.get_default_graph().get_operation_by_name("train_op")
    else:
        return model.train(loss, global_step, name='train_op')

def get_loss(load_model, logits, label):
    if load_model:
        return tf.get_default_graph().get_tensor_by_name(name="loss:0")
    else:
        return model.loss(logits, label, name='loss')

def get_logits(load_model, input):
    if load_model:
        return tf.get_default_graph().get_tensor_by_name("logits:0")
    else:
        return model.inference(input, name='logits')

def get_placeholders(load_model, names=None):

    if load_model:
        g = tf.get_default_graph()
        return g.get_tensor_by_name(names[0]+":0"), \
               g.get_tensor_by_name(names[1]+":0")
    else:
        input_x = tf.placeholder(tf.float32, [None, None, image_height, image_width, input_channels],
                       name=names[0])
        label_y = tf.placeholder(tf.float32, [None, None, image_height, image_width, label_channels],
                                 name=names[1])
        return input_x, label_y

def parse_function(serialized_example):
    """Processes an example ProtoBuf and returns input and label with shape:
        [batch_size, n_frames, height, width, channels]
    """
    #Parse into tensors
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'num_frames': tf.FixedLenFeature([], tf.int64),
            'input': tf.VarLenFeature(tf.string),
            'label': tf.VarLenFeature(tf.string)
        }
    )

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    num_frames = tf.cast(features['num_frames'], tf.int32)

    dense_input = tf.sparse_tensor_to_dense(features['input'], default_value='*')
    dense_label = tf.sparse_tensor_to_dense(features['label'], default_value='*')

    if norm_type:
        in_dtype = tf.float32
    else:
        in_dtype = tf.uint8

    input_list = tf.decode_raw(dense_input, in_dtype)
    label_list = tf.decode_raw(dense_label, tf.uint8)

    #Height and width are dynamically calculated.
    #Therefore, tf.reshape() will cause shape to be undefined and throw an
    #error when running inference for model.
    input_shape = tf.stack([num_frames, height, width, 3])
    label_shape = tf.stack([num_frames, height, width, 1])

    inputs = tf.reshape(input_list, input_shape)
    labels = tf.reshape(label_list, label_shape)

    inputs = tf.cast(inputs, tf.float32)
    labels = tf.cast(labels, tf.float32)

    # label_max = tf.reduce_max(label)

    return inputs, labels


def get_data(load_model, filenames, batch_size, names=None):

    if load_model:
        g = tf.get_default_graph()
        return g.get_tensor_by_name(names[0]+":0"), \
               g.get_tensor_by_name(names[1]+":0"), \
               g.get_operation_by_name(names[2])
    else:
        files = tf.data.Dataset.from_tensor_slices(filenames)
        files = files.shuffle(buffer_size=filenames.__len__())
        train_dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x),
                                   cycle_length=filenames.__len__(), block_length=1)

        # train_dataset = tf.data.TFRecordDataset(filenames)
        # train_dataset = train_dataset.shuffle(buffer_size=40)
        train_dataset = train_dataset.map(parse_function)
        train_dataset = train_dataset.batch(batch_size=batch_size)

        # Runs through tfrecord once. Must call initializer for every epoch
        iterator = train_dataset.make_initializable_iterator()

        input_batch, label_batch = iterator.get_next()

        input_batch = tf.identity(input_batch, name=names[0])
        label_batch = tf.identity(label_batch, name=names[1])

        initializer = iterator.initializer
        # if name:
        #     initializer = tf.identity(iterator.initializer, name)
        # else:
        #     initializer = iterator.initializer

        return input_batch, label_batch, initializer

def train(norm_type, lambda_reg):
    BEST_LOSS = 9999

    """Train frames for a number of steps"""
    print("Global variables before init:")
    for i in tf.global_variables():
        print(i)

    train_filenames = gfile.Glob(train_tfrecords_filename)
    val_filenames = gfile.Glob(val_tfrecords_filename)

    #print("Train filenames:\n", train_filenames)
    #print("Val filenames:\n", val_filenames)

    if load_model_dir:
        model.load(sess, load_model_dir, tags=[tf.saved_model.tag_constants.TRAINING])
        load_model = True
    else:
        load_model = False

    # Necessary in order for input and output to have well defined shapes
    input_x, label_y = get_placeholders(load_model, names=["input_x", "label_y"])

    train_input, train_label, train_initializer = get_data(load_model=load_model,
                                                           filenames=train_filenames,
                                                           batch_size=batch_size,
                                                           names=["train_input", "train_label", "MakeIterator"])

    val_input, val_label, val_initializer = get_data(load_model=load_model,
                                                     filenames=val_filenames,
                                                     batch_size=1,
                                                     names=["val_input", "val_label", "MakeIterator_1"])

    # Gets global_step (i.e. integer that counts how many batches have been processed)
    global_step = tf.train.get_or_create_global_step()

    # Train ops
    logits = get_logits(load_model=load_model, input=input_x)
    model_loss = get_loss(load_model=load_model, logits=logits, label=label_y)
    L2loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ]) * lambda_reg
    print(L2loss)
    loss = model_loss + L2loss
    train_op = get_train_op(load_model=load_model, loss=loss, global_step=global_step)

    # Initialize variables op
    if not load_model:
        epoch_counter = tf.get_variable("epoch_counter", initializer=0, dtype=tf.int32,
                                        trainable=False, use_resource=True)
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        epoch_counter = [v for v in tf.global_variables() if v.name == "epoch_counter:0"][0]

    for v in tf.trainable_variables():
        print(v)
        print (v.shape)
    print("Params: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    # Adds summaries to all trainable variables:
    # for var in tf.trainable_variables():
    #    util.variable_summaries(var)

    merged_summary = get_summary(load_model)

    train_writer = tf.summary.FileWriter(writer_dir, sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    worse_epochs = 0
    val_time = []
    train_time = []

    for epoch in range(num_epochs):
        run_train_epoch(initializer=train_initializer,
                        epoch_counter=epoch_counter,
                        train_op=train_op,
                        loss=loss,
			l2_loss=L2loss,
                        feed_keys=[input_x, label_y],
                        feed_values=[train_input, train_label],
                        train_time=train_time,
                        summary_op=merged_summary,
                        train_writer=train_writer,
                        run_options=run_options,
                        run_metadata=run_metadata)

        if (epoch+1)%val_epochs == 0:
            mse = run_val(initializer=val_initializer,
                    epoch_counter=epoch_counter,
                    logits=logits,
                    loss=loss,
                    feed_keys=[input_x, label_y],
                    feed_values=[val_input, val_label],
                    val_time=val_time)
            if mse < BEST_LOSS:
                BEST_LOSS = mse
                worse_epochs = 0
                model.save(sess, os.path.join(save_model_dir, "best"), overwrite=True,
                           tags=[tf.saved_model.tag_constants.TRAINING])
                with open(val_result_file.split('.')[0] + "_best.txt", "w") as f:
                    f.write("Best results on validation set:\n")
                    f.write("Epoch {}: LOSS = {:.4f}\n".format(epoch, mse))
            else:
                model.save(sess, os.path.join(save_model_dir, "latest"), overwrite=True,
                           tags=[tf.saved_model.tag_constants.TRAINING])
                worse_epochs += 1
                if worse_epochs >= 10:
                    with open(status_file, "a") as f:
                        f.write("Previous {} epochs have shown no improvements.\n"
                                "Stopping training is advised.\n".format(worse_epochs))

        time_left = (np.array(train_time).mean()+np.array(val_time).mean())*(num_epochs-(epoch+1))
        print("Training time left = ", str(datetime.timedelta(seconds=time_left)))


if __name__ == "__main__":

    parser = optparse.OptionParser()

    parser.add_option("-m", "--machine",
                      action="store", type="int", dest="machine",
                      help="Machine index where execution takes place. 0 - local, 1 - neuron0, 2 - neuron1.",
                      default=0)
    parser.add_option("-g", "--gpu",
                      action="store", type="float", dest="gpu",
                      help="Float indicating percent gpu usage. Ranges between [0, 1.0].",
                      default=0)
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
    parser.add_option("-e", "--exp_id",
                      action="store", type="int", dest="experiment",
                      help="Machine index where execution takes place. 0 - local, 1 - neuron0, 2 - neu",
                      default=-1)
    parser.add_option("-l", "--lambda",
                      action="store", type="float", dest="lambda_reg",
                      help="Machine index where execution takes place. 0 - local, 1 - neuron0, 2 - neu",
                      default=-1)

    options, args = parser.parse_args()

    m = options.machine
    if options.norm_type == "ss" or options.norm_type == "fs":
        norm_type = 1
    else:
        norm_type = 0

    val_tfrecord_name = "{}_{}_{}.tfrecords".format("val",
        str(options.norm_type + "_" + options.norm_dim) if norm_type else options.norm_dim,
        options.color_space)
    train_tfrecord_name = "{}_{}/*".format(options.color_space,
        str(options.norm_type + "_" + options.norm_dim) if norm_type else options.norm_dim)

    train_tfrecords_filename = os.path.join(config.train["train_tfrecords_filename"][m], train_tfrecord_name)
    val_tfrecords_filename = os.path.join(config.train["val_tfrecords_filename"][m], val_tfrecord_name)

    load_model_dir = config.train["load_model_dir"][m]    

    base_dir = os.path.join("/home/panda/ic/results", str(options.experiment))
    print(base_dir)

    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
    else:
       print("Experiment ID already exists")   
       exit(1)

    with open(os.path.join(base_dir, "README.md"), "w") as f:
        test_id = "{}_{}\nlambda={}".format(
		options.color_space,
		str(options.norm_type + "_" + options.norm_dim) 
		if norm_type else options.norm_dim, options.lambda_reg)
        f.write(test_id)

    save_model_dir = base_dir
    writer_dir = base_dir
    val_result_file = os.path.join(base_dir, "val_results.txt")
    train_result_file = os.path.join(base_dir, "train_results.txt")
    status_file = os.path.join(base_dir, "status.txt")

    print("Options selected:\n"
          " machine = {}\n"
          " gpu = {}\n"
          " train_dir = {}\n"
          " val_filename = {}\n"
          " colorspace = {}\n"
          " norm_type = {}\n"
          " norm_dim = {}\n"
          " lambda = {}\n".format(m, options.gpu, train_tfrecords_filename, val_tfrecords_filename,
                                  options.color_space, options.norm_type, options.norm_dim, options.lambda_reg))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=options.gpu)

    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        train(norm_type, options.lambda_reg)

