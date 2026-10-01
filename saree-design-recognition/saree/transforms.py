from PIL import Image, ImageOps
import torch
from torchvision import transforms as T

PREPROCESS = {'version': 1, 'size': 224, 'resize': 'aspect_fit_bilinear',
              'padding': [124, 116, 104], 'orientation': 'exif_transpose',
              'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def load_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB').copy()


def fit(im, size=224):
    im = ImageOps.exif_transpose(im).convert('RGB')
    im = ImageOps.contain(im, (size, size), Image.Resampling.BILINEAR)
    out = Image.new('RGB', (size, size), tuple(PREPROCESS['padding']))
    out.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return out


class Preprocess:
    def __init__(self, train=False, color=True, grayscale=False):
        self.train, self.color, self.grayscale = train, color, grayscale
        self.jitter = T.ColorJitter(brightness=.2, contrast=.2, saturation=.6, hue=.2)
        self.normalize = T.Normalize(PREPROCESS['mean'], PREPROCESS['std'])

    def __call__(self, im):
        im = ImageOps.exif_transpose(im).convert('RGB')
        if self.train:
            # No center crop or perspective warp: retain borders and full layout.
            if self.color:
                im = self.jitter(im)
                if torch.rand(()) < .25:
                    im = ImageOps.grayscale(im).convert('RGB')
            # Small padded translation, no mirror that reverses asymmetric motifs.
            im = fit(im)
            dx, dy = torch.randint(-4, 5, (2,)).tolist()
            im = im.transform(im.size, Image.Transform.AFFINE, (1, 0, dx, 0, 1, dy),
                              resample=Image.Resampling.BILINEAR, fillcolor=tuple(PREPROCESS['padding']))
        else:
            im = fit(im)
        if self.grayscale:
            im = ImageOps.grayscale(im).convert('RGB')
        return self.normalize(T.ToTensor()(im))


def recolor(im, seed):
    """Controlled invertible RGB-channel permutation; not a genuine colorway."""
    import numpy as np
    arr = np.asarray(im.convert('RGB'))
    perms = [(1, 2, 0), (2, 0, 1), (2, 1, 0)]
    return Image.fromarray(arr[..., list(perms[seed % len(perms)])])
