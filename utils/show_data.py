import matplotlib.pyplot as plt
import numpy as np


def imshow(img_list):
    k = len(img_list)
    for i in range(k):
        plt.subplot(int(''.join([str(k), '1', str(i+1)])))
        img = img_list[i][0]
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(img_list[i][1])
    plt.show()
