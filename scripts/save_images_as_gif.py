import os
import imageio


def main():
    images_dir = '/home/rogan/git_repos/grokking-experiments/logs/train/runs/2023-09-09_13-45-31/images'
    output_path = '/home/rogan/git_repos/grokking-experiments/logs/train/runs/2023-09-09_13-45-31/fourier_embedding.gif'
    image_filenames = [f for f in os.listdir(images_dir) if f.endswith('.png')]

    image_filenames = sorted([os.path.join(images_dir, f) for f in image_filenames], key=lambda x: int(x.rsplit("_", 1)[1].split(".")[0]))

    # Read in each image file
    images = [imageio.imread(filename) for filename in image_filenames]

    # Save the images as an animated GIF
    imageio.mimsave(output_path, images, duration=100)  # duration is in seconds per frame


if __name__ == "__main__":
    main()
