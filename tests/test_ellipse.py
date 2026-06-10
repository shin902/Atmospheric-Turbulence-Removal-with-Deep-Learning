import cv2
import numpy as np
import pytest

from Modules.ellipse import detect_ellipse, ellipse_to_circle


def draw_ellipse_image(center=(100, 100), axes=(60, 40), angle=0):
    """黒背景に白い楕円を描いた合成画像（axes は半軸長）"""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.ellipse(image, center, axes, angle, 0, 360, (255, 255, 255), -1)
    return image


class TestDetectEllipse:
    def test_合成楕円を検出できる(self):
        image = draw_ellipse_image(center=(100, 100), axes=(60, 40))
        center, axes, angle = detect_ellipse(image)

        assert center is not None
        assert center[0] == pytest.approx(100, abs=5)
        assert center[1] == pytest.approx(100, abs=5)
        # fitEllipse は全軸長（直径相当）を返す。描画時の半軸 (60, 40) → (120, 80)
        assert sorted(axes) == pytest.approx([80, 120], abs=10)

    def test_グレースケール入力でも動く(self):
        image = cv2.cvtColor(draw_ellipse_image(), cv2.COLOR_BGR2GRAY)
        center, axes, angle = detect_ellipse(image)
        assert center is not None

    def test_輪郭がない画像はNoneを返す(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detect_ellipse(image) == (None, None, None)


class TestEllipseToCircle:
    def test_楕円が真円に近づく(self, tmp_path):
        image = draw_ellipse_image(center=(100, 100), axes=(60, 40))
        input_path = tmp_path / "ellipse.jpg"
        cv2.imwrite(str(input_path), image)
        output_path = tmp_path / "circle.jpg"

        transformed = ellipse_to_circle(str(input_path), str(output_path))

        assert transformed is not None
        assert transformed.shape == image.shape
        assert output_path.exists()

        # 変換後の画像から再検出すると軸比がほぼ1（真円）になる
        center, axes, angle = detect_ellipse(transformed)
        assert center is not None
        ratio = max(axes) / min(axes)
        assert ratio == pytest.approx(1.0, abs=0.1)

    def test_読み込めないパスはNoneを返す(self, tmp_path):
        result = ellipse_to_circle(str(tmp_path / "missing.jpg"))
        assert result is None

    def test_楕円検出に失敗したらNoneを返す(self, tmp_path):
        input_path = tmp_path / "black.jpg"
        cv2.imwrite(str(input_path), np.zeros((100, 100, 3), dtype=np.uint8))
        result = ellipse_to_circle(str(input_path))
        assert result is None
