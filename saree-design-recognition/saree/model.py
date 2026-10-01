from pathlib import Path
import io
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from .common import digest
from .transforms import PREPROCESS


class Embedder(nn.Module):
    def __init__(self, kind='trained', pretrained=True):
        super().__init__()
        if kind not in ('trained', 'rgb', 'gray'):
            raise ValueError('kind must be trained, rgb, or gray')
        self.kind = kind
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(576, 256) if kind == 'trained' else nn.Identity()
        self.dim = 256 if kind == 'trained' else 576

    def forward(self, x):
        return F.normalize(self.head(self.pool(self.features(x)).flatten(1)), dim=1)


def save_model(path, model, metadata, optimizer=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save({'state_dict': model.state_dict(), 'kind': model.kind,
                'preprocess': PREPROCESS, 'metadata': metadata,
                'optimizer': optimizer.state_dict() if optimizer else None}, buffer)
    temporary = Path(str(path) + '.tmp')
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def load_model(path, device='cpu'):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}. Run train or baseline first.')
    ck = torch.load(io.BytesIO(Path(path).read_bytes()), map_location='cpu', weights_only=True)
    if ck['preprocess'] != PREPROCESS:
        raise ValueError('Checkpoint preprocessing differs from current implementation')
    model = Embedder(ck['kind'], pretrained=False)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.to(device).eval()
    return model, ck['metadata']


def identity(path):
    return {'checkpoint_sha256': digest(path), 'preprocess': PREPROCESS}
