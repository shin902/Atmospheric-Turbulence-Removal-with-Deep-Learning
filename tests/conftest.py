from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from Modules import noise2noise


class TinyUNet(nn.Module):
    """テスト高速化用の極小モデル。本物のUNetと同じく [-1, 1] のtanh出力を返す"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):
        return torch.tanh(self.conv(x))


@pytest.fixture
def stub_unet(monkeypatch):
    """Denoiser / Noise2Noise が内部で生成する UNet を極小モデルに差し替える"""
    monkeypatch.setattr(noise2noise, "UNet", TinyUNet)
    return TinyUNet


@pytest.fixture
def make_jpg():
    """指定パスに単色のJPG画像を生成するファクトリ"""

    def _make(path, size=(32, 32), color=(128, 64, 32)):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        return path

    return _make
