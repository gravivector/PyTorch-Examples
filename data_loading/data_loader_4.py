from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image


class AutoEncoderDataSet(Dataset):
    def __init__(self, dir_in, dir_gt, transform_in=None, transform_gt=None):
        self.dir_in = self.load_dir_single(dir_in)
        self.dir_gt = self.load_dir_single(dir_gt)
        self.transform_in = transform_in
        self.transform_gt = transform_gt

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

        if self.transform_in:
            img_in = self.transform_in(img_in)
        if self.transform_gt:
            img_gt = self.transform_gt(img_gt)

        return img_in, img_gt


def show_batch(sample_batched, fig_pos, plt_name):
    plt.figure(fig_pos)
    grid = utils.make_grid(sample_batched)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(plt_name)


def main(ps):
    plt.close('all')

    composed = transforms.Compose([transforms.ToTensor()])
    auto_encoder_dataset = AutoEncoderDataSet(ps['DIR_IMG_IN'], ps['DIR_IMG_GT'], composed, composed)
    data_loader = DataLoader(auto_encoder_dataset, batch_size=2, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch)
        if i_batch == 3:
            show_batch(sample_batched[0], 1, 'Input batch from DataLoader')
            show_batch(sample_batched[1], 2, 'Ground truth batch from DataLoader')
            plt.axis('off')
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    ps = {
        'DIR_IMG_IN': 'img/tr/in/',
        'DIR_IMG_GT': 'img/tr/gt/'
    }
    main(ps)

