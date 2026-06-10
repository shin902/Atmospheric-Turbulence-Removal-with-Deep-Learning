import numpy as np
import pytest

from Modules.hsv import HSVImage


@pytest.fixture
def sample_image():
    """左半分がオレンジ（H≈15、デフォルトマスク範囲内）、右半分が青（H=120、範囲外）のBGR画像"""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :2] = (0, 128, 255)  # オレンジ (BGR)
    image[:, 2:] = (255, 0, 0)  # 青 (BGR)
    return image


class TestHSVImage:
    def test_originalはコピーを返す(self, sample_image):
        hsv_img = HSVImage(sample_image)
        sample_image[0, 0] = (0, 0, 0)
        assert tuple(hsv_img.original[0, 0]) == (0, 128, 255)

    def test_hsv変換される(self, sample_image):
        hsv_img = HSVImage(sample_image)
        # 青 (BGR 255,0,0) は OpenCV の HSV で H=120
        assert hsv_img.hsv[0, 3, 0] == 120

    def test_デフォルトマスクはオレンジのみ通す(self, sample_image):
        hsv_img = HSVImage(sample_image)
        mask = hsv_img.get_mask()
        assert mask[0, 0] == 255  # オレンジ
        assert mask[0, 3] == 0  # 青

    def test_カスタム範囲のマスク(self, sample_image):
        hsv_img = HSVImage(sample_image)
        mask = hsv_img.get_mask(hsv_lower=(110, 0, 0), hsv_upper=(130, 255, 255))
        assert mask[0, 0] == 0  # オレンジは範囲外
        assert mask[0, 3] == 255  # 青が範囲内

    def test_マスク適用画像(self, sample_image):
        hsv_img = HSVImage(sample_image)
        hsv_img.get_mask()
        masked = hsv_img.make_masked_image()
        assert tuple(masked[0, 0]) == (0, 128, 255)  # オレンジは残る
        assert tuple(masked[0, 3]) == (0, 0, 0)  # 青は黒に

    def test_マスク未生成でmake_masked_imageはValueError(self, sample_image):
        hsv_img = HSVImage(sample_image)
        with pytest.raises(ValueError, match="マスクが生成されていません"):
            hsv_img.make_masked_image()

    def test_明示的にマスクを渡せる(self, sample_image):
        hsv_img = HSVImage(sample_image)
        mask = np.zeros((4, 4), dtype=np.uint8)  # 全部マスクアウト
        masked = hsv_img.make_masked_image(mask)
        assert masked.sum() == 0

    def test_hsvセッターはサイズ違いを拒否(self, sample_image):
        hsv_img = HSVImage(sample_image)
        with pytest.raises(ValueError):
            hsv_img.hsv = np.zeros((8, 8, 3), dtype=np.uint8)

    def test_hsvセッターは同サイズなら受け付ける(self, sample_image):
        hsv_img = HSVImage(sample_image)
        new_hsv = np.zeros((4, 4, 3), dtype=np.uint8)
        hsv_img.hsv = new_hsv
        assert hsv_img.hsv.sum() == 0
