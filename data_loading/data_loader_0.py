from torch.utils.data import Dataset
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image


class AutoEncoderDataSet(Dataset):
    def __init__(self, dir_in, dir_gt):
        self.dir_in = [join(dir_in, x) for x in listdir(dir_in) if self.is_image_file(x)]
        self.dir_gt = [join(dir_gt, x) for x in listdir(dir_gt) if self.is_image_file(x)]

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

    def load_img(self, filename):
        img = Image.open(filename)
        return img

    def __len__(self):
        return len(self.dir_in)

    def __getitem__(self, index):
        img_in = self.load_img(self.dir_in[index])
        img_gt = self.load_img(self.dir_gt[index])

        return img_in, img_gt


def main(ps):
    plt.close('all')
    f, axarr = plt.subplots(4, 2)
    auto_encoder_dataset = AutoEncoderDataSet(ps['DIR_IMG_IN'], ps['DIR_IMG_GT'])
    for i in range(len(auto_encoder_dataset)):
        img_in, img_gt = auto_encoder_dataset[i]

        axarr[i, 0].imshow(img_in)
        axarr[i, 0].set_title('Input image #{}'.format(i))
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(img_gt)
        axarr[i, 1].set_title('Ground truth image #{}'.format(i))
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
