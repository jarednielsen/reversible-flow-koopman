from datasets import BouncingMNIST, MovingSymbols, GolfSwing, GolfSwingClips
from scipy.misc import imsave, imresize
import numpy as np
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def save_bouncing_mnist(n_sequences, image_size, sequence_length):
    data = MovingSymbols(size=n_sequences, height=image_size, width=image_size, sequence_length=sequence_length)
    X, y = data[0] # X has shape (sequence_length, 3, height, width)

    for i_video in range(len(data)):
        if i_video % 100 == 0:
            print(i_video)
        X, y = data[i_video]
        dir_path = '../data/bouncing_mnist_size{}_seqlen{}_test/video_{}'.format(image_size, sequence_length, i_video)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        for i_frame in range(len(X)):
            frame_path = '{0}/image_{1:0=2d}.png'.format(dir_path, i_frame) # print '01, 02, ... 14'
            im = np.array(np.transpose(X[i_frame], (1, 2, 0)))
            im = imresize(im, 4.0)
            imsave(frame_path, im)



# data_old = GolfSwing()
# data_new = GolfSwingClips()
# seq_old = data_old[0][0]
# seq_new = data_new[0][0]
# print("old shape: {}, {}".format(seq_old.shape, type(seq_old)))
# print("new shape: {}, {}".format(seq_new.shape, type(seq_new)))

# img_old = seq_old[0]
# img_new = seq_new[0]

# print(img_new.shape)

# # imsave("temp.png", img_new)
# imsave("temp.png", img_new)


# plt.imshow(np.transpose(data_old.denormalize(img_old[0]), [1, 2, 0]))
# plt.imshow(img_new[0])
# plt.savefig

# import pdb; pdb.set_trace()

# data = BouncingMNIST()


save_bouncing_mnist(n_sequences=20000, image_size=64, sequence_length=15)