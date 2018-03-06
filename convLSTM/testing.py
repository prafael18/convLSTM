import tensorflow as tf

load_model_dir = "/home/rafael/Documents/unicamp/ic/src/save"

num_epochs = 5
batch_size = 1
image_height = 135
image_width = 240
input_channels = 3
label_channels = 1

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, tags=['train'], export_dir=load_model_dir)
    kernel = sess.run('conv2d/kernel:0')
    bias = sess.run('conv2d/bias:0')
    global_step = sess.run(tf.train.get_or_create_global_step())
    print("kernel = {} \nbias = {}".format(kernel,bias))
    print("global_step = {}".format(global_step))
    global_step_init = tf.Graph.get_operation_by_name(tf.get_default_graph(),
                                                      name='global_step/Assign')
    sess.run(global_step_init)
    print(global_step_init)
    global_step = sess.run(tf.train.get_or_create_global_step())
    print("global_step after init = {}".format(global_step))

    for var in tf.global_variables():
        print(var)

    for op in tf.get_default_graph().get_operations():
        print(str(op.name))