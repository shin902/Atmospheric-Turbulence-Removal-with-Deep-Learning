# Noise2Noise 使い方ガイド

`src/Modules/noise2noise.py` は、Noise2Noise（U-Net）による画像のノイズ除去モジュールです。
コマンドラインから **学習** と **ノイズ除去（推論）** の両方を実行できます。

## できること

| やりたいこと | 使うコマンド |
|---|---|
| 学習済みモデルで画像のノイズを除去したい | `denoise` |
| フォルダ内の画像をまとめてノイズ除去したい | `denoise`（`--input` にフォルダを指定） |
| 自分のデータでモデルを学習させたい | `train` |
| 中断した学習を続きから再開したい | `train --resume` |

## 事前準備

[README のセットアップ手順](../README.md#セットアップ方法) に従って `uv sync` まで完了させてください。
以降のコマンドはすべて **リポジトリのルートフォルダ** で実行する前提です。

---

## ノイズ除去（推論）

学習済みモデル（`.pth` ファイル）があれば、学習データは一切不要です。

### 画像1枚を処理する

```bash
uv run python src/Modules/noise2noise.py denoise \
  --model "Resources/AI/model_dir/fixed_model.pth" \
  --input "input.jpg" \
  --output "denoised.jpg"
```

### フォルダ内の画像をまとめて処理する

`--input` にフォルダを指定すると、その中の `.jpg` をすべて処理して `--output` フォルダに同名で保存します。

```bash
uv run python src/Modules/noise2noise.py denoise \
  --model "Resources/AI/model_dir/fixed_model.pth" \
  --input "Resources/Images/19_57_44" \
  --output "Resources/Input and Output/denoised"
```

処理の進み具合は `[3/120] 003.jpg` のように表示されます。

### denoise のオプション一覧

| オプション | 必須 | 説明 |
|---|---|---|
| `--model` | ✅ | 学習済みモデル（`.pth`）のパス |
| `--input` | ✅ | 入力画像ファイル、または jpg が入ったフォルダ |
| `--output` | ✅ | 出力ファイル名、またはフォルダ（フォルダは自動で作られます） |
| `--device` | | `cuda` / `mps` / `cpu` を指定。省略すると自動選択 |

---

## 学習

### データの置き方

学習用・検証用フォルダの中に「連番画像が入ったサブフォルダ」を置いてください。
サブフォルダ内のファイル名順に隣り合う2枚が、自動的に学習ペアになります（N枚 → N-1ペア）。

```
Resources/AI/
├── train_data/          ← 学習用
│   ├── 19_57_44/        ← 撮影セッションごとのフォルダ
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── 20_13_05/
│       └── ...
└── valid_data/          ← 検証用（構成は train_data と同じ）
    └── ...
```

### 学習を実行する

```bash
uv run python src/Modules/noise2noise.py train \
  --train-dir "Resources/AI/train_data" \
  --valid-dir "Resources/AI/valid_data" \
  --model-dir "Resources/AI/model_dir" \
  --epochs 10 \
  --run-name my_model
```

実行すると `--model-dir` に2つのファイルが保存されます。

- `my_model.pth` — 検証ロスが最も良かった時点のモデル（ノイズ除去にそのまま使えます）
- `my_model.csv` — エポックごとの学習・検証ロスの記録

### 中断した学習を再開する

```bash
uv run python src/Modules/noise2noise.py train \
  --train-dir "Resources/AI/train_data" \
  --valid-dir "Resources/AI/valid_data" \
  --model-dir "Resources/AI/model_dir" \
  --epochs 10 \
  --run-name my_model \
  --resume my_model.pth
```

`--resume` には `--model-dir` 内のファイル名か、フルパスを指定できます。

再開時はチェックポイントから「これまでのベスト検証ロス」と「損失の履歴」も復元されます。
そのため、再開後に前回より悪いモデルでベストモデルが上書きされることはなく、
損失CSVにも再開前からの全エポックの履歴が記録されます。

> ⚠️ 旧バージョンで学習した `.pth`（ベストロス情報を含まない形式）から再開した場合は、
> ベスト判定がリセットされるため、再開後の1エポック目で必ずモデルが上書き保存されます。
> 旧モデルから再開するときは `--run-name` を変えて、元のファイルを残すことをおすすめします。

### train のオプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--train-dir` | `../../Resources/AI/train_data` | 学習データのフォルダ |
| `--valid-dir` | `../../Resources/AI/valid_data` | 検証データのフォルダ |
| `--model-dir` | `../../Resources/AI/model_dir` | モデルとCSVの保存先 |
| `--epochs` | `10` | 学習エポック数 |
| `--max-pairs` | `200` | 使う画像ペア数の上限。`0` で無制限 |
| `--run-name` | `fixed_model` | 保存ファイル名（拡張子なし） |
| `--device` | 自動選択 | `cuda` / `mps` / `cpu` |
| `--resume` | なし | 再開元のモデルファイル |

> ⚠️ デフォルトのパスは `src/Modules` フォルダから実行した場合の相対パスです。
> リポジトリのルートから実行する場合は、上の例のように `--train-dir` などを明示的に指定してください。

---

## Python から使う（開発者向け）

### ノイズ除去だけしたい場合 — `Denoiser`

学習データの指定は不要です。

```python
from Modules.noise2noise import Denoiser

denoiser = Denoiser()  # デバイスは自動選択（cuda → mps → cpu）
denoiser.load_model("Resources/AI/model_dir/fixed_model.pth")  # フルパスでOK
denoiser.denoise_image("input.jpg", "output.jpg")
```

`model_dir` を渡しておくと、ファイル名だけでモデルを読み込めます。

```python
denoiser = Denoiser(model_dir="Resources/AI/model_dir")
denoiser.load_model("fixed_model.pth")  # model_dir からの相対名
```

### 学習したい場合 — `Noise2Noise`

```python
from Modules.noise2noise import Noise2Noise

trainer = Noise2Noise(
    train_dir="Resources/AI/train_data",
    valid_dir="Resources/AI/valid_data",
    model_dir="Resources/AI/model_dir",
    max_pairs=200,
)
trainer.train(epochs=10, run_name="my_model")
```

`Noise2Noise` は `Denoiser` を継承しているので、学習後にそのまま `denoise_image()` も呼べます。
データセットの読み込みは `train()` を呼んだ時に初めて行われるため、推論だけなら `train_dir` / `valid_dir` は省略できます。

---

## よくある質問

**Q. ノイズ除去だけしたいのに学習データのフォルダが要求される？**
A. 旧バージョンの仕様です。現在は `Denoiser` クラス、または CLI の `denoise` コマンドを使えば学習データは不要です。

**Q. GPU がなくても動く？**
A. 動きます。CUDA（NVIDIA GPU）→ MPS（Apple Silicon）→ CPU の順で自動的に選択されます。CPU では学習にかなり時間がかかるため、推論のみの利用を推奨します。

**Q. jpg 以外の画像は使える？**
A. 学習データとフォルダ一括処理は `.jpg` のみ対象です。1枚ずつの `denoise` であれば PNG など Pillow が読める形式も入力できます（出力形式は `--output` の拡張子で決まります）。

**Q. 学習の途中経過はどこで見られる？**
A. 実行中はターミナルに10バッチごとのロスが表示されます。終了後は `--model-dir` に保存される CSV（`run-name.csv`）で全エポックの推移を確認できます。
