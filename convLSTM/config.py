
train = {
    "train_tfrecords_filename": "/home/storage_local/rafael_ic/data/train/train1_of_1.tfrecords",
    "val_tfrecords_filename": "/home/storage_local/rafael_ic/data/val/val1_of_1.tfrecords",
    "save_model_dir": "/home/storage_local/rafael_ic/save",
    "load_model_dir": None,
    # "load_model_dir": "/home/rafael/Documents/unicamp/ic/src/save",
    "writer_dir": "/home/storage_local/rafael_ic/log",
    "num_epochs": 500,
    "batch_size": 5,
    "image_height": 135,
    "image_width": 240,
    "input_channels": 3,
    "label_channels": 1,
    "val_epochs": 5,
    "val_result_file": "/home/storage_local/rafael_ic/log/val_results.txt"
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
