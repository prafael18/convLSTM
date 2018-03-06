# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

import input
import model
import config
import util
import infer

tfrecords_filename = config.train["tfrecords_filename"]
save_model_dir = config.train["save_model_dir"]
load_model_dir = config.train["load_model_dir"]
num_epochs = config.train["num_epochs"]
batch_size = config.train["batch_size"]
image_height = config.train["image_height"]
image_width = config.train["image_width"]
input_channels = config.train["input_channels"]
label_channels = config.train["label_channels"]
writer_dir = config.train["writer_dir"]

def train():
    """Train frames for a number of steps"""

    with tf.Session() as sess:

        print("Global variables before init:")
        for i in tf.global_variables():
            print(i)

        if load_model_dir is None:

            train_dataset = tf.data.TFRecordDataset(tfrecords_filename)
            train_dataset = train_dataset.shuffle(buffer_size=40)

            train_dataset = train_dataset.map(input.parse_function)
            train_dataset = train_dataset.batch(batch_size=batch_size)

            #Runs through tfrecord once. Must call initializer for every epoch
            iterator = train_dataset.make_initializable_iterator()

            input_batch, label_batch = iterator.get_next()

            #Necessary in order for input and output to have well defined shapes
            input_x = tf.placeholder_with_default(input_batch,
                                                  [batch_size, None, image_height, image_width, input_channels],
                                                  name="input_x")
            label_y = tf.placeholder_with_default(label_batch,
                                                  [batch_size, None, image_height, image_width, label_channels],
                                                  name="label_y")

            print(label_y)
            # Gets global_step (i.e. integer that counts how many batches have been processed)
            global_step = tf.train.get_or_create_global_step()

            # Train ops
            logits = model.inference(input_x, name='logits')
            loss = model.loss(logits, label_y, name='loss')
            train_op = model.train(loss, global_step, name='train_op')

            # Initialize variables op
            init = tf.global_variables_initializer()
            sess.run(init)
            iterator_init = iterator.initializer

            #Adds summaries to all trainable variables:
            for var in tf.trainable_variables():
                util.variable_summaries(var)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(writer_dir)

        else:
            model.load(sess, load_model_dir, tags=[tf.saved_model.tag_constants.TRAINING])

            #Reference to current graph
            g = tf.get_default_graph()

            #Iterator op that should be initialized every epoch
            iterator_init = tf.Graph.get_operation_by_name(g, name="MakeIterator")

            #Retrieves tensor associated to loss op.
            loss = g.get_tensor_by_name(name='loss:0')

            #Retrieves train_op that should be passed to sess.run()
            train_op = g.get_operation_by_name(name='train_op')

            #Summary op
            merged = g.get_tensor_by_name(name="Merged/MergeSummary:0")

        train_writer = tf.summary.FileWriter(writer_dir, sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        print("Global variables after init:")
        for i in tf.global_variables():
            print(i)

        for epoch in range(num_epochs):
            sess.run(iterator_init)
            batch_loss = []
            start_time = time.time()

            while True:
                try:
                    _, loss_val, summary = sess.run([train_op, loss, merged],
                                                    options=run_options,
                                                    run_metadata=run_metadata)
                except tf.errors.OutOfRangeError:
                    train_writer.add_run_metadata(run_metadata, "Epoch {}".format(epoch))
                    train_writer.add_summary(summary, epoch)
                    batch_loss.append(loss_val)
                    print("Epoch {} completed in {} seconds.\n"
                          "Average cross-entropy loss is: {}"
                          .format(epoch + 1, time.time() - start_time,
                                  np.mean(np.array(batch_loss))))
                    model.save(sess, save_model_dir, overwrite=True,
                               tags=[tf.saved_model.tag_constants.TRAINING])
                    break


if __name__ == "__main__":
    train()