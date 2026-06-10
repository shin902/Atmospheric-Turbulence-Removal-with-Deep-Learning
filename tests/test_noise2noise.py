import argparse
import csv
from pathlib import Path

import pytest
import torch

from Modules import noise2noise
from Modules.noise2noise import (
    Denoiser,
    Noise2Noise,
    NoisyDataset,
    UNet,
    _build_parser,
    _prompt,
    _prompt_optional,
    _run_denoise,
    _run_interactive,
    _run_name_type,
    main,
    resolve_device,
)


class TestResolveDevice:
    def test_明示指定はそのまま使われる(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_自動選択でcudaが最優先(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device(None) == torch.device("cuda")

    def test_cudaがなければmps(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert resolve_device(None) == torch.device("mps")

    def test_どちらもなければcpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert resolve_device(None) == torch.device("cpu")


class TestRunNameType:
    def test_単なるファイル名は通る(self):
        assert _run_name_type("my_model") == "my_model"

    def test_パス区切りを含むと拒否される(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _run_name_type("subdir/model")

    def test_絶対パスも拒否される(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _run_name_type("/tmp/model")


class TestBuildParser:
    def test_trainのデフォルト値(self):
        args = _build_parser().parse_args(["train"])
        assert args.command == "train"
        assert args.epochs == 10
        assert args.max_pairs == 200
        assert args.run_name == "fixed_model"
        assert args.device is None
        assert args.resume is None

    def test_train_run_nameにパス区切りはエラー(self, capsys):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["train", "--run-name", "a/b"])

    def test_denoiseは必須引数が揃えばパースできる(self):
        args = _build_parser().parse_args(
            ["denoise", "--model", "m.pth", "--input", "in.jpg", "--output", "out.jpg"]
        )
        assert args.command == "denoise"
        assert args.model == "m.pth"

    def test_denoiseはmodel必須(self, capsys):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(
                ["denoise", "--input", "in.jpg", "--output", "out.jpg"]
            )

    def test_サブコマンド必須(self, capsys):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])


class TestNoisyDataset:
    def test_存在しないディレクトリはエラー(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NoisyDataset(tmp_path / "missing")

    def test_ペアがなければエラー(self, tmp_path):
        (tmp_path / "empty_folder").mkdir()
        with pytest.raises(FileNotFoundError):
            NoisyDataset(tmp_path)

    def test_ルート直下の画像も連続ペア化される(self, tmp_path, make_jpg):
        make_jpg(tmp_path / "a.jpg")
        make_jpg(tmp_path / "b.jpg")
        make_jpg(tmp_path / "c.jpg")
        dataset = NoisyDataset(tmp_path)
        assert len(dataset) == 2

    def test_ルート直下とサブフォルダの両方が使われる(self, tmp_path, make_jpg):
        make_jpg(tmp_path / "a.jpg")
        make_jpg(tmp_path / "b.jpg")
        for i in range(3):
            make_jpg(tmp_path / "folder1" / f"{i:03d}.jpg")
        dataset = NoisyDataset(tmp_path)
        assert len(dataset) == 3  # ルート直下: 1ペア + folder1: 2ペア

    def test_N枚からN_1ペアできる(self, tmp_path, make_jpg):
        for i in range(3):
            make_jpg(tmp_path / "folder1" / f"{i:03d}.jpg")
        dataset = NoisyDataset(tmp_path)
        assert len(dataset) == 2

    def test_max_pairsで上限が効く(self, tmp_path, make_jpg):
        for i in range(4):
            make_jpg(tmp_path / "folder1" / f"{i:03d}.jpg")
        dataset = NoisyDataset(tmp_path, max_pairs=1)
        assert len(dataset) == 1

    def test_getitemは正規化済みテンソルを返す(self, tmp_path, make_jpg):
        for i in range(2):
            make_jpg(tmp_path / "folder1" / f"{i:03d}.jpg")
        dataset = NoisyDataset(tmp_path)
        input_tensor, target_tensor = dataset[0]
        # size=(780, 972) は PIL の (width, height) なのでテンソルは (C, H, W) = (3, 972, 780)
        assert input_tensor.shape == (3, 972, 780)
        assert target_tensor.shape == (3, 972, 780)
        assert input_tensor.min() >= -1.0
        assert input_tensor.max() <= 1.0


class TestDenoiserResolveModelPath:
    def test_model_dir未設定ならそのまま(self, stub_unet):
        denoiser = Denoiser(device="cpu")
        assert denoiser._resolve_model_path("model.pth") == Path("model.pth")

    def test_単なるファイル名はmodel_dir相対(self, stub_unet, tmp_path):
        denoiser = Denoiser(device="cpu", model_dir=tmp_path)
        assert denoiser._resolve_model_path("model.pth") == tmp_path / "model.pth"

    def test_パス区切りを含むならそのまま(self, stub_unet, tmp_path):
        denoiser = Denoiser(device="cpu", model_dir=tmp_path)
        assert denoiser._resolve_model_path("sub/model.pth") == Path("sub/model.pth")

    def test_絶対パスはそのまま(self, stub_unet, tmp_path):
        denoiser = Denoiser(device="cpu", model_dir=tmp_path)
        abs_path = tmp_path / "other" / "model.pth"
        assert denoiser._resolve_model_path(str(abs_path)) == abs_path


class TestSaveLoadModel:
    def test_保存して読み込める(self, stub_unet, tmp_path):
        trainer = Noise2Noise(model_dir=tmp_path, device="cpu")
        trainer.best_valid_loss = 0.123
        trainer.train_losses = [1.0, 0.5]
        trainer.valid_losses = [0.9, 0.4]
        trainer.save_model("model.pth")
        assert (tmp_path / "model.pth").exists()

        loaded = Noise2Noise(model_dir=tmp_path, device="cpu")
        loaded.load_model("model.pth")
        assert loaded.best_valid_loss == pytest.approx(0.123)
        assert loaded.train_losses == [1.0, 0.5]
        assert loaded.valid_losses == [0.9, 0.4]
        for p1, p2 in zip(trainer.model.parameters(), loaded.model.parameters()):
            assert torch.equal(p1, p2)

    def test_旧形式チェックポイントでも読み込める(self, stub_unet, tmp_path):
        trainer = Noise2Noise(model_dir=tmp_path, device="cpu")
        # best_valid_loss などのキーを持たない旧形式を再現
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
            },
            tmp_path / "old.pth",
        )
        loaded = Noise2Noise(model_dir=tmp_path, device="cpu")
        loaded.load_model("old.pth")
        assert loaded.best_valid_loss == float("inf")
        assert loaded.train_losses == []
        assert loaded.valid_losses == []

    def test_推論専用Denoiserでも読み込める(self, stub_unet, tmp_path):
        trainer = Noise2Noise(model_dir=tmp_path, device="cpu")
        trainer.save_model("model.pth")

        denoiser = Denoiser(device="cpu")
        checkpoint = denoiser.load_model(str(tmp_path / "model.pth"))
        assert "model_state_dict" in checkpoint

    def test_深いmodel_dirも自動作成される(self, stub_unet, tmp_path):
        deep_dir = tmp_path / "a" / "b" / "c"
        Noise2Noise(model_dir=deep_dir, device="cpu")
        assert deep_dir.is_dir()


class TestDenoiseImage:
    def test_出力画像が生成される(self, stub_unet, tmp_path, make_jpg):
        input_path = make_jpg(tmp_path / "input.jpg", size=(16, 16))
        output_path = tmp_path / "output.jpg"

        denoiser = Denoiser(device="cpu")
        denoiser.denoise_image(str(input_path), str(output_path))

        assert output_path.exists()
        from PIL import Image

        with Image.open(output_path) as img:
            assert img.size == (16, 16)
            assert img.mode == "RGB"


class TestNoise2NoiseTraining:
    def test_model_dirなしのtrainはエラー(self, stub_unet):
        trainer = Noise2Noise(device="cpu")
        with pytest.raises(ValueError):
            trainer.train(epochs=1)

    def test_データディレクトリなしの_setup_dataはエラー(self, stub_unet, tmp_path):
        trainer = Noise2Noise(model_dir=tmp_path, device="cpu")
        with pytest.raises(ValueError):
            trainer._setup_data()

    def test_1エポック学習でモデルとCSVが保存される(self, stub_unet, tmp_path, make_jpg):
        train_dir = tmp_path / "train"
        valid_dir = tmp_path / "valid"
        model_dir = tmp_path / "models"
        for i in range(3):
            make_jpg(train_dir / "f1" / f"{i:03d}.jpg")
        for i in range(2):
            make_jpg(valid_dir / "f1" / f"{i:03d}.jpg")

        trainer = Noise2Noise(
            train_dir=train_dir,
            valid_dir=valid_dir,
            model_dir=model_dir,
            device="cpu",
        )
        trainer.train(epochs=1, run_name="test_run")

        assert (model_dir / "test_run.pth").exists()
        assert (model_dir / "test_run.csv").exists()
        assert len(trainer.train_losses) == 1
        assert len(trainer.valid_losses) == 1

    def test_save_losses_to_csvの内容(self, stub_unet, tmp_path):
        trainer = Noise2Noise(model_dir=tmp_path, device="cpu")
        trainer.train_losses = [1.0, 0.5]
        trainer.valid_losses = [0.9, 0.4]
        trainer.save_losses_to_csv("losses.csv")

        with open(tmp_path / "losses.csv", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["Train Loss", "Valid Loss"]
        assert rows[1] == ["1.0", "0.9"]
        assert rows[2] == ["0.5", "0.4"]


class FakeDenoiser:
    """_run_denoise のCLIロジック検証用。生成された全インスタンスを記録する"""

    instances = []

    def __init__(self, device=None):
        self.loaded_model = None
        self.denoise_calls = []
        FakeDenoiser.instances.append(self)

    def load_model(self, path):
        self.loaded_model = path

    def denoise_image(self, input_path, output_path):
        self.denoise_calls.append((input_path, output_path))
        Path(output_path).write_bytes(b"fake")


@pytest.fixture
def fake_denoiser(monkeypatch):
    FakeDenoiser.instances = []
    monkeypatch.setattr(noise2noise, "Denoiser", FakeDenoiser)
    return FakeDenoiser


def _denoise_args(model, input_path, output_path):
    return argparse.Namespace(
        model=str(model), input=str(input_path), output=str(output_path), device=None
    )


class TestRunDenoise:
    def test_ディレクトリ指定で大文字小文字混在の拡張子も処理される(
        self, fake_denoiser, tmp_path, make_jpg
    ):
        input_dir = tmp_path / "input"
        make_jpg(input_dir / "a.jpg")
        make_jpg(input_dir / "b.JPG")
        make_jpg(input_dir / "c.jpeg")
        make_jpg(input_dir / "d.png")  # 対象外
        output_dir = tmp_path / "output"

        _run_denoise(_denoise_args("m.pth", input_dir, output_dir))

        denoiser = fake_denoiser.instances[0]
        assert denoiser.loaded_model == "m.pth"
        processed = sorted(Path(inp).name for inp, _ in denoiser.denoise_calls)
        assert processed == ["a.jpg", "b.JPG", "c.jpeg"]
        assert output_dir.is_dir()

    def test_対象画像がないディレクトリはエラー(self, fake_denoiser, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            _run_denoise(_denoise_args("m.pth", input_dir, tmp_path / "out"))

    def test_単一ファイルで出力先の親ディレクトリが作成される(
        self, fake_denoiser, tmp_path, make_jpg
    ):
        input_path = make_jpg(tmp_path / "input.jpg")
        output_path = tmp_path / "nested" / "dir" / "output.jpg"

        _run_denoise(_denoise_args("m.pth", input_path, output_path))

        assert output_path.exists()


class TestUNet:
    def test_出力形状は入力と同じでtanh範囲(self):
        model = UNet()
        model.eval()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 32, 32)
        assert y.min() >= -1.0
        assert y.max() <= 1.0


def _set_inputs(monkeypatch, responses):
    it = iter(responses)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


class TestPrompt:
    def test_空入力でデフォルトを返す(self, monkeypatch):
        _set_inputs(monkeypatch, [""])
        assert _prompt("質問", default="abc") == "abc"

    def test_入力値をそのまま返す(self, monkeypatch):
        _set_inputs(monkeypatch, ["hello"])
        assert _prompt("質問") == "hello"

    def test_castで数値に変換する(self, monkeypatch):
        _set_inputs(monkeypatch, ["5"])
        assert _prompt("質問", default=10, cast=int) == 5

    def test_デフォルトなしの空入力は再入力させる(self, monkeypatch):
        _set_inputs(monkeypatch, ["", "value"])
        assert _prompt("質問") == "value"

    def test_不正な数値は再入力させる(self, monkeypatch, capsys):
        _set_inputs(monkeypatch, ["abc", "3"])
        assert _prompt("質問", cast=int) == 3
        assert "数値を入力してください" in capsys.readouterr().out


class TestPromptOptional:
    def test_空入力はNone(self, monkeypatch):
        _set_inputs(monkeypatch, [""])
        assert _prompt_optional("質問") is None

    def test_入力値をそのまま返す(self, monkeypatch):
        _set_inputs(monkeypatch, ["cuda"])
        assert _prompt_optional("質問") == "cuda"


class TestInteractiveTrain:
    def test_入力値からNamespaceを作って_run_trainを呼ぶ(self, monkeypatch, tmp_path):
        _set_inputs(monkeypatch, [
            str(tmp_path / "train"),
            str(tmp_path / "valid"),
            str(tmp_path / "model"),
            "5",
            "100",
            "my_model",
            "cpu",
            "",
        ])
        captured = {}
        monkeypatch.setattr(noise2noise, "_run_train", lambda args: captured.setdefault("args", args))

        noise2noise._interactive_train()

        args = captured["args"]
        assert args.train_dir == str(tmp_path / "train")
        assert args.valid_dir == str(tmp_path / "valid")
        assert args.model_dir == str(tmp_path / "model")
        assert args.epochs == 5
        assert args.max_pairs == 100
        assert args.run_name == "my_model"
        assert args.device == "cpu"
        assert args.resume is None

    def test_run_nameにパス区切りを入れると再入力になる(self, monkeypatch, tmp_path, capsys):
        _set_inputs(monkeypatch, [
            str(tmp_path / "train"),
            str(tmp_path / "valid"),
            str(tmp_path / "model"),
            "1",
            "1",
            "a/b",
            "valid_name",
            "",
            "",
        ])
        captured = {}
        monkeypatch.setattr(noise2noise, "_run_train", lambda args: captured.setdefault("args", args))

        noise2noise._interactive_train()

        assert captured["args"].run_name == "valid_name"
        assert "パス区切り" in capsys.readouterr().out


class TestInteractiveDenoise:
    def test_入力値からNamespaceを作って_run_denoiseを呼ぶ(self, monkeypatch, tmp_path):
        _set_inputs(monkeypatch, [
            str(tmp_path / "model.pth"),
            str(tmp_path / "input.jpg"),
            str(tmp_path / "output.jpg"),
            "",
        ])
        captured = {}
        monkeypatch.setattr(noise2noise, "_run_denoise", lambda args: captured.setdefault("args", args))

        noise2noise._interactive_denoise()

        args = captured["args"]
        assert args.model == str(tmp_path / "model.pth")
        assert args.input == str(tmp_path / "input.jpg")
        assert args.output == str(tmp_path / "output.jpg")
        assert args.device is None


class TestRunInteractive:
    def test_train選択で_interactive_trainが呼ばれる(self, monkeypatch):
        _set_inputs(monkeypatch, ["train"])
        called = []
        monkeypatch.setattr(noise2noise, "_interactive_train", lambda: called.append("train"))
        monkeypatch.setattr(noise2noise, "_interactive_denoise", lambda: called.append("denoise"))

        _run_interactive()

        assert called == ["train"]

    def test_denoise選択で_interactive_denoiseが呼ばれる(self, monkeypatch):
        _set_inputs(monkeypatch, ["denoise"])
        called = []
        monkeypatch.setattr(noise2noise, "_interactive_train", lambda: called.append("train"))
        monkeypatch.setattr(noise2noise, "_interactive_denoise", lambda: called.append("denoise"))

        _run_interactive()

        assert called == ["denoise"]

    def test_不正な入力は再度プロンプトされる(self, monkeypatch, capsys):
        _set_inputs(monkeypatch, ["foo", "denoise"])
        monkeypatch.setattr(noise2noise, "_interactive_denoise", lambda: None)

        _run_interactive()

        assert "'train' か 'denoise' を入力してください。" in capsys.readouterr().out


class TestMain:
    def test_引数なしはinteractiveになる(self, monkeypatch):
        called = []
        monkeypatch.setattr(noise2noise, "_run_interactive", lambda: called.append("interactive"))
        main([])
        assert called == ["interactive"]

    def test_interactiveコマンドを明示しても同じ(self, monkeypatch):
        called = []
        monkeypatch.setattr(noise2noise, "_run_interactive", lambda: called.append("interactive"))
        main(["interactive"])
        assert called == ["interactive"]

    def test_trainコマンドは_run_trainを呼ぶ(self, monkeypatch):
        called = []
        monkeypatch.setattr(noise2noise, "_run_train", lambda args: called.append(args))
        main(["train"])
        assert len(called) == 1
        assert called[0].command == "train"

    def test_denoiseコマンドは_run_denoiseを呼ぶ(self, monkeypatch, tmp_path):
        called = []
        monkeypatch.setattr(noise2noise, "_run_denoise", lambda args: called.append(args))
        main(["denoise", "--model", "m.pth", "--input", "in.jpg", "--output", "out.jpg"])
        assert len(called) == 1
        assert called[0].command == "denoise"
