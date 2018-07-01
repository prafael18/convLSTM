from scipy.misc import imread, imsave, imshow
import sys
import os
from skimage import io
from skimage import transform as skt
from skimage import filters as skf
import numpy as np
import glob
import cv2
import random
import optparse
import time
from datetime import timedelta

def _get_rng(rng):
    if not isinstance(rng, (list, tuple)):
        rng = (rng, rng)
    return rng

def _rot90(arr, reps=1):
    """
    Performs 90 degrees rotation 'reps' times.
    Assumes image with shape ([n_samples, n_channels,] height, width).
    """
    for __ in range(reps%4):
        arr = arr.swapaxes(-2, -1)[..., ::-1]
    return arr

def rot90(x, y, reps=1):
    x, y = _rot90(x, reps), y if y is None else _rot90(y, reps)
    return x, y

def _hmirr(img):
    """
    Flips image horizontally.
    Assumes image with shape ([n_samples, n_channels,] height, width).
    """
    return img[..., ::-1]

def hmirr(x, y):
    x, y = _hmirr(x), y if y is None else _hmirr(y)
    return x, y

def some_of(x, y=None, ops=[]):
    """
    Chooses one operation from ops.
    """
    op = np.random.choice(ops)
    x = op(x)
    if y is not None:
        y = op(y)
    return x, y

def _rotation(img, angle):
    """
    Rotates image in degrees in counter-clockwise direction.
    Assumes image in [0, 1] with shape ([n_samples, n_channels,] height, width).
    """
    img = skt.rotate(img, angle=angle, resize=False, mode="constant",
        preserve_range=True).astype(img.dtype)
    return img

def rotation(x, y, rng, fixed, isRNG):
    if isRNG:
        angle = np.random.uniform(*rng)
    else:
        angle = fixed
    x = _rotation(x, angle)
    y = y if y is None else _rotation(y, angle)
    return x, y

def _shear(img, shear):
    """
    Shears image.
    Assumes image in [0, 1] with shape ([n_samples, n_channels,] height, width).
    """
    at = skt.AffineTransform(shear=shear)
    img = skt.warp(img, at)
    return img

def shear(x, y, rng, fixed, isRNG):
    if isRNG:
        shear = np.random.uniform(*rng)
    else:
        shear = fixed
    x, y = _shear(x, shear), y if y is None else _shear(y, shear)
    return x, y

def _translation(img, transl):
    """
    Performs shift in image in dx, dy = transl.
    Assumes image in [0, 1] with shape ([n_samples, n_channels,] height, width).
    """
    at = skt.AffineTransform(translation=transl)
    # img = img.swapaxes(0, 1).swapaxes(1, 2)
    img = skt.warp(img, at)
    # img = img.swapaxes(2, 1).swapaxes(1, 0)
    return img

def translation(x, y, rng, fixed, isRNG):
    h, w = x.shape[-2:]
    if isRNG:
        transl = (int(np.random.uniform(*rng)*w), int(np.random.uniform(*rng)*h))
    else:
        transl = (fixed*w, fixed*h)
    x, y = _translation(x, transl), y if y is None else _translation(y, transl)
    return x, y

def _add_noise(img, noise):
    """
    Adds noise to image.
    Assumes image in [0, 1].
    """
    img = img + noise
    return img

def add_noise(x, y, rng, fixed, isRNG):
    if isRNG:
        noise = np.random.uniform(*rng, size=x.shape).astype("float32")
    else:
        noise = fixed
    x, y = _add_noise(x, noise), y
    return x, y

def _mul_noise(img, noise):
    """
    Multiplies image by a factor.
    Assumes image in [0, 1].
    """
    img = img*noise
    return img

def mul_noise(x, y, rng, fixed, isRNG):
    if isRNG:
        noise = np.random.uniform(*rng)
    else:
        noise = fixed

    x, y = _mul_noise(x, noise), y
    return x, y

def _blur(img, sigma):
    """
    Applies gaussian blur to image.
    Assumes image in [0, 1] with shape ([n_samples, n_channels,] height, width).
    """
    # img = img.swapaxes(0, 1).swapaxes(1, 2)
    for i in range(img.shape[-1]):
        img[..., i] = skf.gaussian(img[..., i], sigma=sigma)
    # img = img.swapaxes(2, 1).swapaxes(1, 0)
    return img

def blur(x, y, rng, fixed, isRNG):
    if isRNG:
        sigma = np.random.uniform(*rng)
    else:
        sigma = fixed
    x, y = _blur(x, sigma), y
    return x, y

def identity(x, y):
    return x, y

def _unit_norm(img, minn, maxx, dtype="float32"):
    img = ((img - minn)/max(maxx - minn, 1)).astype(dtype)
    return img

def _unit_denorm(img, minn, maxx, dtype="float32"):
    img = (img*(maxx - minn) + minn).astype(dtype)
    return img

#mapping of strings to methods
OPS_MAP = {
    "rot90": rot90,
    "rotation": rotation,
    "shear": shear,
    "translation": translation,
    "add_noise": add_noise,
    "mul_noise": mul_noise,
    "blur": blur,
    "identity": identity,
    "hmirr": hmirr,
}

def augment(xy, op_seqs, apply_on_y=False, add_iff_op=True):
    """
    Performs data augmentation on x, y sample.

    op_seqs is a list of sequences of operations.
    Each sequence must be in format (op_name, op_prob, op_kwargs).
    Example of valid op_seqs:
    [
        [
            ('identity', 1.0, {}),
        ],
        [
            ('hmirr', 1.0, {}),
            ('rot90', 1.0, {'reps': 3})
        ],
        [
            ('rotation', 0.5, {'rng': (-10, 10)}),
        ]
    ]
    ('identity' is necessary to keep the original image in the returned list.)

    add_iff_op: adds image to augm list only if some operation happened.
    """
    #list of augmented images
    augm = []

    #pre-processing x, y for augmentation
    x, y = xy
    x_minn, x_maxx, x_dtype = x.min(), x.max(), x.dtype
    x = _unit_norm(x, x_minn, x_maxx, "float32")
    if apply_on_y:
        y_minn, y_maxx, y_dtype = y.min(), y.max(), y.dtype
        y = _unit_norm(y, y_minn, y_maxx, "float32")

    #applying sequences
    for op_seq in op_seqs:
        _x, _y = x.copy(), y.copy() if apply_on_y else None

        some_op = False
        #applying sequence of operations
        for name, prob, kwargs in op_seq:
            op = OPS_MAP[name]
            if np.random.uniform(0.0, 1.0) <= prob:
                some_op = True
                _x, _y = op(_x, _y, **kwargs)

        #adding sample to augm list
        if some_op or not add_iff_op:
            _x = _unit_denorm(_x, x_minn, x_maxx, x_dtype)
            if apply_on_y:
                _y = _unit_denorm(_y, y_minn, y_maxx, y_dtype)
            augm.append((_x, _y if apply_on_y else y))

    return augm

#data augmentation default operation sequence to be applied in about every op
_all_augm_ops = [
    ("blur", 1.0, {"rng": (0.5, 1.0), "fixed": 0.75, "isRNG": False}),
    ("translation", 1.0, {"rng": (-0.1, 0.1), "fixed": 0.0, "isRNG": False}),
    ("rotation", 1.0, {"rng": (-35, 35), "fixed": 0, "isRNG": False}),
    ("shear", 1.0, {"rng": (-0.1, 0.1), "fixed": 0.0, "isRNG": False}),
    ("add_noise", 1.0, {"rng": (-0.1, 0.1), "fixed": 0.0, "isRNG": False}),
    ("mul_noise", 1.0, {"rng": (0.9, 1.1), "fixed": 1.0, "isRNG": False}),
]

def save_video(x_fp, y_fp, name, x_save_dir, y_save_dir, rng):

    x_cap = cv2.VideoCapture(x_fp)
    x_fps = int(x_cap.get(cv2.CAP_PROP_FPS))

    width = int(x_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(x_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    y_cap = cv2.VideoCapture(y_fp)
    y_fps = int(y_cap.get(cv2.CAP_PROP_FPS))

    #Define random ops for clip
    num_ops = random.randrange(1, len(_all_augm_ops))

    index_list = []
    for _ in range(num_ops):
        index_list.append(random.randrange(0, len(_all_augm_ops)))
    index_list = set(index_list)
    _def_augm_ops = [_all_augm_ops[i] for i in index_list]

    for op_name, _, ops in _def_augm_ops:
        if rng:
            ops["isRNG"] = True
        else:
            ops["isRNG"] = False
            if op_name == "add_noise":
                ops["fixed"] = np.random.uniform(ops["rng"][0], ops["rng"][1], size=(height, width, 3)).astype("float32")
            else:
                ops["fixed"] = np.random.uniform(ops["rng"][0], ops["rng"][1])

    _augment_op_seqs = [
        [("identity", 1.0, {}),] + _def_augm_ops,
        [("hmirr", 1.0, {}),] + _def_augm_ops,
    ]

    x_out = []
    y_out = []
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for op in _augment_op_seqs:
        op_name = op[0][0]
        x_out_fp = os.path.join(x_save_dir, name+"_"+op_name+".avi")
        y_out_fp = os.path.join(y_save_dir, name+"_"+op_name+".avi")
        x_out.append(cv2.VideoWriter(x_out_fp, fourcc, x_fps, (width, height)))
        y_out.append(cv2.VideoWriter(y_out_fp, fourcc, y_fps, (width, height)))

    x_ret, x_frame = x_cap.read()
    y_ret, y_frame = y_cap.read()
    while x_ret and y_ret:
        xy = (x_frame, y_frame)

        aug_xy = augment(xy, _augment_op_seqs, apply_on_y=True)

        for (i, (x, y)) in enumerate(aug_xy):
            x_out[i].write(x)
            y_out[i].write(y)

        x_ret, x_frame = x_cap.read()
        y_ret, y_frame = y_cap.read()

    x_cap.release()
    y_cap.release()
    for i in range(x_out.__len__()):
        x_out[i].release()
        y_out[i].release()

if __name__ == "__main__":
    #base_dir = "/home/rafael/Documents/ic/src/data/"
    base_dir = "/home/panda/raw_data"
    rng = False

    # Parse options
    parser = optparse.OptionParser()
    parser.add_option("-r", "--rng",
                      action="store_true", dest="rng",
                      help="Paramaters for augmentation functions are determined randomly per frame.",
                      default=False)
    parser.add_option("-d", "--dest_dir",
                      action="store", type="string", dest="dest_dir",
                      help="Destination directory for augmented videos",
                      default="augm_out")

    options, args = parser.parse_args()

    # Get cmdline args
    rng = options.rng
    dest_dir = options.dest_dir

    input_dir = os.path.join(base_dir, "train")
    x_dir = os.path.join(input_dir, "inputs", "*.avi")
    y_dir = os.path.join(input_dir, "labels", "*.avi")

    out_dir = os.path.join(base_dir, dest_dir)
    x_out_dir = os.path.join(out_dir, "inputs")
    y_out_dir = os.path.join(out_dir, "labels")

    if not os.path.isdir(y_out_dir): os.makedirs(y_out_dir)
    if not os.path.isdir(x_out_dir): os.makedirs(x_out_dir)


    x_fps = glob.glob(x_dir)
    y_fps = glob.glob(y_dir)

    xy_fps = [(x_fps[i], y_fps[i]) for i in range(x_fps.__len__())]

    print("Processing a total of {} videos".format(len(xy_fps)))
    total_vids = len(xy_fps)
    vid_num = 0
    for x, y in xy_fps:
        init_time = time.time()
        bn = os.path.basename(x).split('.')[0]
        save_video(x_fp=x, y_fp=y, name=bn, x_save_dir=x_out_dir, y_save_dir=y_out_dir, rng=rng)
        vid_num += 1
        print("Finished processing video {}/{}. Estimated time left: {}."
                .format(vid_num, total_vids, str(timedelta(seconds=(total_vids-vid_num)*(time.time()-init_time)))))
