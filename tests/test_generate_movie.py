import cv2

from Modules.generate_movie import generate_movie


class TestGenerateMovie:
    def test_jpg列から動画が生成される(self, tmp_path, make_jpg):
        input_dir = tmp_path / "frames"
        for i in range(3):
            make_jpg(input_dir / f"{i:03d}.jpg", size=(64, 48))
        movie_path = tmp_path / "out.mp4"

        generate_movie(str(input_dir), str(movie_path))

        assert movie_path.exists()
        assert movie_path.stat().st_size > 0

        cap = cv2.VideoCapture(str(movie_path))
        try:
            assert cap.isOpened()
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
            assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
            assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
        finally:
            cap.release()
