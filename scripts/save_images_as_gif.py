import os
import imageio
import numpy as np
from collections import defaultdict


def main():
    images_dir = '/home/rogan/git_repos/grokking-experiments/logs/train/runs/2023-10-11_08-19-43/images'
    output_path = '/home/rogan/git_repos/grokking-experiments/images/animation.gif'

    subdirs = ['embedding', 'attention', 'activation']
    image_filenames = defaultdict(list)
    for subdir in subdirs:
        image_filenames[subdir] = [f for f in os.listdir(os.path.join(images_dir, subdir)) if f.endswith('.png')]
        image_filenames[subdir] = sorted([os.path.join(images_dir, subdir, f) for f in image_filenames[subdir]], key=lambda x: int(x.rsplit("_", 1)[1].split(".")[0]))

    max_len = max([len(image_filenames[subdir]) for subdir in subdirs])
    for subdir in subdirs:
        image_filenames[subdir] = image_filenames[subdir][:max_len]

    # Read in each image file
    images = []
    for i in range(max_len):
        image = []
        for subdir in subdirs:
            image.append(imageio.imread(image_filenames[subdir][i]))
        images.append(np.concatenate(image, axis=1))

    # Save the images as an animated GIF
    imageio.mimsave(output_path, images, duration=100)  # duration is in seconds per frame


if __name__ == "__main__":
    main()
