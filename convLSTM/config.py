
train = {
    "machine_index": 2,

    "train_tfrecords_filename":
        ["/home/rafael/Documents/unicamp/ic/src/data/train/tfr",
         "/home/panda/ic/data/train",
         "/home/storage_local/panda/data/train",
         "/home/rafael/Documents/ic/src/data/train/tfr"],

    "val_tfrecords_filename":
        ["/home/rafael/Documents/unicamp/ic/src/data/val/tfr",
         "/home/panda/ic/data/val",
         "/home/storage_local/panda/data/val",
         "/home/rafael/Documents/ic/src/data/val/tfr"],

    "save_model_dir":
        ["/home/rafael/Documents/unicamp/ic/src/save",
         "/home/panda/ic/save",
         "/home/storage_local/panda/save",
         "/home/rafael/Documents/ic/src/save"],

    "load_model_dir":
        [None,
         None,
         None,
         None],

    "result_dir":
        [None,
         "/home/panda/ic/results",
         None,
         None],

    "writer_dir":
        ["/home/rafael/Documents/unicamp/ic/src/log",
         "/home/panda/ic/log",
         "/home/storage_local/panda/log",
         "/home/rafael/Documents/ic/src/log"],

    "val_result_file":
        ["/home/rafael/Documents/unicamp/ic/src/log/val_results.txt",
         "/home/panda/ic/log/val_results.txt",
         "/home/storage_local/panda/log/val_results.txt",
         "/home/rafael/Documents/ic/src/log/val_results.txt"],

    "train_result_file":
        ["/home/rafael/Documents/unicamp/ic/src/log/train_results.txt",
         "/home/panda/ic/log/train_results.txt",
         "/home/storage_local/panda/log/train_results.txt",
         "/home/rafael/Documents/ic/src/log/train_results.txt"],

    "status_file":
        ["/home/rafael/Documents/unicamp/ic/src/log/status.txt",
         "/home/panda/ic/log/status.txt",
         "/home/storage_local/panda/log/status.txt",
         "/home/rafael/Documents/ic/src/log/status.txt"],

    "num_epochs": 100,
    "batch_size": 5,
    "image_height": 135,
    "image_width": 240,
    "input_channels": 3,
    "label_channels": 1,
    "val_epochs": 1,
    "lambda": 5e-3
}



eval = {
    "tfrecords_filename": "/home/rafael/Documents/unicamp/ic/src/data/test/tfr/test1_of_1.tfrecords",
    "load_model_dir": "/home/rafael/Documents/unicamp/ic/src/save",
    "num_epochs": 1,
    "batch_size": 1,
    "image_height": 135,
    "image_width": 240,
    "input_channels": 3,
    "label_channels": 1
}
