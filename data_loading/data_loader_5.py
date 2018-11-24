import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import join
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        img_in, img_gt = sample['img_in'], sample['img_gt']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img_in = np.asarray(img_in)
        img_in = img_in.transpose((2, 0, 1))
        img_gt = np.asarray(img_gt)
        img_gt = img_gt.transpose((2, 0, 1))
        return {'img_in': torch.from_numpy(img_in), 'img_gt': torch.from_numpy(img_gt)}


class RandomRotate(object):
    """Rotate the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being rotated. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            sample {img_in PIL Image, img_gt PIL Image}: Images to be rotated.
        Returns:
            {img_in PIL Image, img_gt PIL Image}: Randomly rotated images.
        """
        img_in, img_gt = sample['img_in'], sample['img_gt']

        if random.random() < self.p:
            angle = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
            img_in = img_in.transpose(angle)
            img_gt = img_gt.transpose(angle)

        return {'img_in': img_in, 'img_gt': img_gt}


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            sample {img_in PIL Image, img_gt PIL Image}: Images to be flipped.
        Returns:
            {img_in PIL Image, img_gt PIL Image}: Randomly flipped image.
        """
        img_in, img_gt = sample['img_in'], sample['img_gt']

        if random.random() < self.p:
            img_in = img_in.transpose(Image.FLIP_TOP_BOTTOM)
            img_gt = img_gt.transpose(Image.FLIP_TOP_BOTTOM)

        return {'img_in': img_in, 'img_gt': img_gt}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            sample {img_in PIL Image, img_gt PIL Image}: Images to be flipped.
        Returns:
            {img_in PIL Image, img_gt PIL Image}: Randomly flipped image.
        """
        img_in, img_gt = sample['img_in'], sample['img_gt']

        if random.random() < self.p:
            img_in = img_in.transpose(Image.FLIP_LEFT_RIGHT)
            img_gt = img_gt.transpose(Image.FLIP_LEFT_RIGHT)

        return {'img_in': img_in, 'img_gt': img_gt}


class RandomCrop(object):
    """Crop the given PIL Images randomly."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        Args:
            sample {img_in PIL Image, img_gt PIL Image}: Images to be cropped.
        Returns:
            {img_in PIL Image, img_gt PIL Image}: Randomly cropped image.
        """
        img_in, img_gt = sample['img_in'], sample['img_gt']

        w, h = img_in.size
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img_in = img_in.crop((left, top, left + new_w, top + new_h))
        img_gt = img_gt.crop((left, top, left + new_w, top + new_h))

        return {'img_in': img_in, 'img_gt': img_gt}


class AutoEncoderDataSet(Dataset):
    def __init__(self, dir_in, dir_gt, transform=None):
        self.dir_in = self.load_dir_single(dir_in)
        self.dir_gt = self.load_dir_single(dir_gt)
        self.transform = transform

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"])

    def load_img(self, filename):
        img = Image.open(filename)
        return img

    def load_dir_single(self, directory):
        return [join(directory, x) for x in listdir(directory) if self.is_image_file(x)]

    def __len__(self):
        return len(self.dir_in)

    def __getitem__(self, index):
        img_in = self.load_img(self.dir_in[index])
        img_gt = self.load_img(self.dir_gt[index])
        sample = {'img_in': img_in, 'img_gt': img_gt}

        if self.transform:
            sample = self.transform(sample)

        return sample


def show_batch(sample_batched, fig_pos, plt_name):
    plt.figure(fig_pos)
    grid = utils.make_grid(sample_batched)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(plt_name)


def main(ps):
    plt.close('all')

    composed = transforms.Compose(
        [RandomCrop(128), RandomHorizontalFlip(), RandomVerticalFlip(), RandomRotate(), ToTensor()])
    auto_encoder_dataset = AutoEncoderDataSet(ps['DIR_IMG_IN'], ps['DIR_IMG_GT'], composed)
    data_loader = DataLoader(auto_encoder_dataset, batch_size=5, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, sample_batched['img_in'].size(), sample_batched['img_gt'].size())
        if i_batch == 1:
            show_batch(sample_batched['img_in'], 1, 'Input batch from DataLoader')
            show_batch(sample_batched['img_gt'], 2, 'Ground truth batch from DataLoader')
            plt.axis('off')
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    ps = {
        'DIR_IMG_IN': 'img/tr/in/',
        'DIR_IMG_GT': 'img/tr/gt/'
    }
    main(ps)
