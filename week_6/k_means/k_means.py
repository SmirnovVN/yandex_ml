import pandas as pd
import numpy as np
from skimage import img_as_float
from sklearn.cluster import KMeans
from skimage.io import imread
from matplotlib import pyplot as plt


def main():
    image = img_as_float(imread('parrots.jpg'))
    colours = image.reshape(-1, 3)

    i = 0
    for i in range(9, 20):
        cls = KMeans(init='k-means++', random_state=241, n_clusters=i)
        cls.fit(colours)
        clusters = pd.DataFrame(np.column_stack([cls.labels_, colours[:, 0], colours[:, 1], colours[:, 2]]))
        clusters.columns = ['labels', 'r', 'g', 'b']
        medians = clusters.groupby(['labels']).median().values

        img_by_centers = np.array([cls.cluster_centers_[x] for x in cls.labels_]).reshape(image.shape)
        plt.imshow(img_by_centers)
        plt.savefig(str(i)+'-centers.png')

        mse_centers = ((image - img_by_centers) ** 2).sum() / (image.shape[0] * image.shape[1] * 3)
        psnr_centers = 10 * np.log10(1 / mse_centers)
        print('centers', psnr_centers)
        if psnr_centers > 20:
            break

        img_by_medians = np.array([medians[x] for x in cls.labels_]).reshape(image.shape)
        plt.imshow(img_by_medians)
        plt.savefig(str(i)+'-medians.png')

        mse_medians = ((image - img_by_medians) ** 2).sum() / (image.shape[0] * image.shape[1] * 3)
        psnr_medians = 10 * np.log10(1 / mse_medians)
        print('medians', psnr_medians)
        if psnr_medians > 20:
            break

    with open('task_1.txt', 'w') as f:
        f.write(str(i))


if __name__ == '__main__':
    main()
