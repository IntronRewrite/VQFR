import random
import time
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from torchvision import transforms

from vqfr.data.transforms import augment
from vqfr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from vqfr.utils.registry import DATASET_REGISTRY
from vqfr.data.data_util import paired_paths_from_folder,paths_from_folder


@DATASET_REGISTRY.register()
class FFHQDataset(data.Dataset):
    """FFHQ dataset for StyleGAN.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.

    """

    def __init__(self, opt):
        super(FFHQDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']
        transforms.Resize((320, 480), antialias=True)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', " f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # FFHQ has 70000 images in total
            self.paths = paths_from_folder(self.gt_folder)
            # self.paths = [osp.join(self.gt_folder, f'norain-{v+1:d}.png') for v in range(10230)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)

        img_gt = self.resize(img_gt)

        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        return {'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
