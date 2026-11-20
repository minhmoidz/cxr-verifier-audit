import os
import numpy as np
from PIL import Image
from utils import utils
from torch.utils import data
import torchvision.transforms as transforms


class SiameseDataset(data.Dataset):
    def __init__(self, phase='training', n_channels=3, image_size=256, b=None, m=None, eps=None, image_path='./',
                 blur_sigma=None, elastic_mu=None, train_pairing='sanitized_sanitized'):
        """The dataset that is used for the patient verification experiments.

        :param phase: str
            Specifies the phase.
            Training: phase='training'
            Validation: phase='validation'
            Testing: phase='testing'
        :param n_channels: int
            Specifies the number of channels for the input images. Default: n_channels=3
        :param image_size: int
            The image size to be used. Default: image_size=256
        :param b: int
            Specifies the size of the grid cells for DP pixelization.
        :param m: int
            Specifies the m-neighborhood (DP sensitivity factor).
        :param eps: int
            Specifies the DP privacy budget (smaller values indicate greater privacy).
        :param image_path: str
            The path to the folder where the images are stored.
        :param blur_sigma: float
            Standard deviation of a Gaussian blur, in pixels at `image_size`. Added by this
            audit as a non-learned, monotonically tunable sanitizer, so that a comparison can be
            made at utility matched to the deformation family rather than only against controls
            that have destroyed most of the image. Wired exactly like DP-Pix: applied to side A
            in every phase and to side B only during training and validation (subject to
            `train_pairing`), which preserves the test-time asymmetry between a released record
            and a real one.
        :param train_pairing: str
            Controls whether side B (the second endpoint of a pair) is sanitized during
            training/validation, for every perturbation type applied inside this dataset
            (DP-Pix, Gaussian blur, random elastic). It has no effect at test time, which is
            always query-sanitized / reference-raw by construction, independent of this flag.

            'sanitized_sanitized' (default): side B is sanitized during training and validation,
            reproducing the released protocol's train/validation pairing exactly.

            'sanitized_raw': side B is left untouched during training and validation too, so the
            attacker trains on the same query-sanitized / reference-raw pairing it is tested on.

            An explicit two-value enum rather than a boolean, so a config cannot silently mean
            'sanitize side B always' vs 'never' without saying which regime it names.
        """

        if train_pairing not in ('sanitized_sanitized', 'sanitized_raw'):
            raise ValueError(
                "train_pairing must be 'sanitized_sanitized' or 'sanitized_raw', got "
                + repr(train_pairing))

        self.phase = phase
        self.n_channels = n_channels
        self.image_size = image_size
        self.PATH = image_path
        self.train_pairing = train_pairing

        # Load the image pairs (filenames)
        if self.phase == 'training':
            train_pair_file = os.environ.get('TRAIN_PAIRS_FILE', './image_pairs/image_pairs_training_10000.txt')
            self.image_pairs = np.loadtxt(train_pair_file, dtype=str)
        elif self.phase == 'validation':
            self.image_pairs = np.loadtxt(os.environ.get('VAL_PAIRS_FILE', './image_pairs/image_pairs_validation_2000.txt'), dtype=str)
        elif self.phase == 'testing':
            self.image_pairs = np.loadtxt('./image_pairs/image_pairs_testing_5000.txt', dtype=str)
        else:
            raise Exception('Invalid argument for parameter phase!')

        # Parameters that are used for the DP-Pix experiments (if self.use_dp_pix is True)
        self.use_dp_pix = b is not None and m is not None and eps is not None
        self.elastic_mu = elastic_mu
        self.blur_sigma = blur_sigma
        if blur_sigma is not None:
            k = int(2 * np.ceil(3 * blur_sigma) + 1)
            self.blur = transforms.GaussianBlur(kernel_size=k, sigma=float(blur_sigma))
        self.b = b
        self.m = m
        self.eps = eps

        # Define transformations
        if self.n_channels == 1:
            self.transform = transforms.ToTensor()
        elif self.n_channels == 3:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.resize = transforms.Resize((self.image_size, self.image_size))
        self.to_pil = transforms.ToPILImage()

        self.images_1 = []
        self.images_2 = []
        self.labels = []

        for i in range(len(self.image_pairs)):
            # Load images
            image1 = pil_loader(self.PATH + self.image_pairs[i][0], self.n_channels)
            image2 = pil_loader(self.PATH + self.image_pairs[i][1], self.n_channels)

            # Resize images
            image1 = self.resize(image1)
            image2 = self.resize(image2)

            # Apply DP-Pix if respective flag is activated
            if self.use_dp_pix:
                image1 = self.transform(image1).unsqueeze(0)
                image1 = utils.dp_pix(image_tensor=image1, b=self.b, m=self.m, eps=self.eps, plot=False).squeeze(0)
                image1 = self.to_pil(image1)

                if (self.phase == 'training' or self.phase == 'validation') and self.train_pairing == 'sanitized_sanitized':
                    image2 = self.transform(image2).unsqueeze(0)
                    image2 = utils.dp_pix(image_tensor=image2, b=self.b, m=self.m, eps=self.eps, plot=False).squeeze(0)
                    image2 = self.to_pil(image2)

            # Apply Gaussian blur if requested, with the same side asymmetry as DP-Pix
            if self.blur_sigma is not None:
                image1 = self.blur(image1)
                if (self.phase == 'training' or self.phase == 'validation') and self.train_pairing == 'sanitized_sanitized':
                    image2 = self.blur(image2)


            # Append images to respective lists
            self.images_1.append(image1)
            self.images_2.append(image2)
            self.labels.append(float(self.image_pairs[i][2]))

        # Random elastic deformation (PROTOCOL_ADDENDUM section C), same side asymmetry as
        # DP-Pix and blur. Applied here in batches rather than per image inside the loop: the
        # 9x9 smoothing convolution dominates, and batching it is ~30x faster over 17k images.
        # The field is still seeded per filename, so the result is identical either way.
        if self.elastic_mu is not None:
            names1 = [self.image_pairs[i][0] for i in range(len(self.image_pairs))]
            self.images_1 = self._elastic_batch(self.images_1, names1)
            if (self.phase == 'training' or self.phase == 'validation') and self.train_pairing == 'sanitized_sanitized':
                names2 = [self.image_pairs[i][1] for i in range(len(self.image_pairs))]
                self.images_2 = self._elastic_batch(self.images_2, names2)

    def _elastic(self, pil_image, filename):
        t = self.transform(pil_image).unsqueeze(0)
        t = utils.elastic_warp(t, self.elastic_mu, [filename])
        return self.to_pil(t.squeeze(0).clamp(0, 1))

    def _elastic_batch(self, images, names, chunk=64):
        import torch as _t
        out = []
        for i in range(0, len(images), chunk):
            batch = _t.stack([self.transform(im) for im in images[i:i + chunk]])
            warped = utils.elastic_warp(batch, self.elastic_mu, names[i:i + chunk])
            out += [self.to_pil(w.clamp(0, 1)) for w in warped]
        return out

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        # Get image1 and image2
        image1 = self.images_1[index]
        image2 = self.images_2[index]

        # Apply the transformation
        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return image1, image2, self.labels[index]


def pil_loader(path, n_channels):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 1:
            return img.convert('L')
        elif n_channels == 3:
            return img.convert('RGB')
        else:
            raise ValueError('Invalid value for parameter n_channels!')
