import numpy as np

def sim(list, pred, label):
    # Pre-process data:
    # (1) Normalize pred and label between 0 and 1
    # (2) Make sure that all pixel values add up to 1
    # print("Sum pred = ", np.sum(pred))

    pred = (pred - np.min(pred))/(np.max(pred)-np.min(pred))
    pred = pred/np.sum(pred)
    label = label/np.sum(label)
    sim_coeff = np.minimum(pred, label)
    sim_list = [np.sum(s) for s in sim_coeff]
    list.append(np.mean(np.array(sim_list)))
    return


def cc(cc_list, pred, label):
    # Pred and label have shapes (batch_size, frames, height, width, channels)
    warnings.simplefilter("error", RuntimeWarning)

    num_videos = pred.shape[0]
    num_frames = pred.shape[1]

    corr_coeff = []
    for v in range(num_videos):
        for f in range(num_frames):
            try:
                # Normalize data to have mean 0 and variance 1
                pred = (pred - np.mean(pred)) / np.std(pred)
                label = (label - np.mean(label)) / np.std(label)

                # Calculate correlation coefficient for every frame
                pd = pred - np.mean(pred)
                ld = label - np.mean(label)
                corr_coeff.append((pd * ld).sum() / np.sqrt((pd * pd).sum() * (ld * ld).sum()))
            except RuntimeWarning:
                pass
            # print("Failed to append frame {} to corr_coeff".format(i))
        cc_list.append(np.mean(np.array(corr_coeff)))
    return


def mse(mse_list, pred, label):

    num_videos = pred.shape[0]
    num_frames = pred.shape[1]

    mean_squared_error = []

    for v in range(num_videos):
        for f in range(num_frames):
           mean_squared_error.append(np.mean((pred-label)**2))
    mse_list.append(np.mean(np.array(mean_squared_error)))


if __name__ == "__main__":
    mkimages()
    calculate_metrics()