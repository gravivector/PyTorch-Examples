from torch.utils.data import Dataset
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image


class AutoEncoderDataSet(Dataset):
    def __init__(self, dir_in, dir_gt):
        self.dir_in = self.load_dir_single(dir_in)
        self.dir_gt = self.load_dir_single(dir_gt)

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

        return sample


def main(ps):
    plt.close('all')
    f, axarr = plt.subplots(4, 2)
    auto_encoder_dataset = AutoEncoderDataSet(ps['DIR_IMG_IN'], ps['DIR_IMG_GT'])
    for i in range(len(auto_encoder_dataset)):
        sample = auto_encoder_dataset[i]
        img_in, img_gt = sample['img_in'], sample['img_gt']

        axarr[i, 0].imshow(img_in)
        axarr[i, 0].set_title('Input image #{}'.format(i))
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(img_gt)
        axarr[i, 1].set_title('Ground truth #{}'.format(i))
        axarr[i, 1].axis('off')

        if i == 3:
            f.subplots_adjust(hspace=0.5)
            plt.show()
            break


if __name__ == "__main__":
    ps = {
        'DIR_IMG_IN': 'img/tr/in/',
        'DIR_IMG_GT': 'img/tr/gt/'
    }
    main(ps)




