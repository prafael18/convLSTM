
train = {
    "tfrecords_filename": "/home/rafael/Documents/unicamp/ic/src/data/val/tfr/val1_of_1.tfrecords",
    "save_model_dir": "/home/rafael/Documents/unicamp/ic/src/save",
    "load_model_dir": None,
    "writer_dir": "/home/rafael/Documents/unicamp/ic/src/log/train",
    "num_epochs": 10,
    "batch_size": 1,
    "image_height": 135,
    "image_width": 240,
    "input_channels": 3,
    "label_channels": 1
}

eval = {
    "tfrecords_filename": "/home/rafael/Documents/unicamp/ic/src/data/val/tfr/val1_of_1.tfrecords",
    "load_model_dir": "/home/rafael/Documents/unicamp/ic/src/save",
    "num_epochs": 1,
    "batch_size": 1,
    "image_height": 135,
    "image_width": 240,
    "input_channels": 3,
    "label_channels": 1
}