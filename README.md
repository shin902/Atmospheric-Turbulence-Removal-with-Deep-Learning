# 大気のゆらぎをAIを使用し補正
- 木星の画像（動画）を、CNN、U-net、Noise2Noise、affine変換を使用して大気の揺らぎを補正するアプリを、Pythonをメインで開発しています。

## セットアップ方法

このプロジェクトはパッケージ管理ツール **uv** を使用しています。  
以下の手順に従って環境を構築してください。

### 1. uv のインストール

uv は公式のインストールスクリプトを使って簡単にインストールできます。

**Windows（PowerShell）の場合:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux の場合:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール後、ターミナル（またはコマンドプロンプト）を再起動してください。  
`uv --version` と入力してバージョンが表示されれば成功です。

### 2. リポジトリの取得（2つの方法）

#### 方法A: git clone を使う（推奨）

```bash
git clone https://github.com/shin902/Atmospheric-Turbulence-Removal-with-Deep-Learning.git
cd Atmospheric-Turbulence-Removal-with-Deep-Learning
```

#### 方法B: Download ZIP を使う（Git未導入向け）

1. GitHubのリポジトリページを開く  
2. 緑色の **Code** ボタンをクリック  
3. **Download ZIP** を選択して保存  
4. ZIPを展開して、展開先フォルダへ移動

### 3. 依存関係のインストール

クローンしたフォルダ内で以下のコマンドを実行すると、必要なパッケージが自動でインストールされます。

```bash
uv sync
```

これだけで環境構築は完了です。

### 4. アプリケーションの起動（例）

GUI を起動する場合:
```bash
uv run python src/GUI/GUI.py
```

### 5. Windows版セットアップ補足（PowerShell）

Windows では PowerShell で以下を順番に実行すると確実です。

```powershell
# uv のインストール（未インストールの場合のみ）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# リポジトリの取得
git clone https://github.com/shin902/Atmospheric-Turbulence-Removal-with-Deep-Learning.git
cd Atmospheric-Turbulence-Removal-with-Deep-Learning

# 依存関係のセットアップ
uv sync
```

### 6. CLI 実行方法（学習・生成）

このプロジェクトは `uv run python ...` で各スクリプトをCLI実行できます。

#### 学習（Noise2Noise）

`src/Modules/noise2noise.py` の末尾にある `if __name__ == "__main__":` ブロックで学習できます。  
学習を実行する場合は、このブロック内の以下のコメントアウトを外してください。

```python
# trainer.train(epochs=1000)
```

その後、次のコマンドを実行します。

```bash
uv run python src/Modules/noise2noise.py
```

※ `noise2noise.py` の `train_dir` / `valid_dir` はサンプル値です。  
実行ディレクトリとデータ配置に合わせて、実在するパスに変更してから実行してください。

#### 生成（推論）

ノイズ除去付きの連番画像生成＋動画化:

```bash
uv run python src/main/movie_denoise.py
```

アフィン補正画像の生成:

```bash
uv run python src/main/movie_affine.py
```

実行前に、各スクリプト内のパス設定を手元のデータ配置に合わせて変更してください。
- `src/main/movie_denoise.py`: `img_folder`, `out_folder`, `movie_path`
- `src/main/movie_affine.py`: `input_dir`, `output_dir`

## 使用技術
### 開発関係
- Python
    - pytorch
    - numpy
    - opencv
    - flet(GUI実装)
- uv

## uv(仮想環境とパッケージ管理)のおおまかな使い方
| コマンド   | 説明   | 用途・タイミング   | 例   |
|------------|----------------------------------------------------------------------|---------------------------------------------------|-------------------------------------|
| `uv sync`  | `pyproject.toml` に基づいて依存関係を同期し、ロックファイルを生成。 | 初回セットアップ、依存の更新時                   | `uv sync`                           |
| `uv run`   | 仮想環境内でコマンドを実行。pythonファイルを後ろにくっつけると、そのファイルを実行                                         | 依存をインストールせずにスクリプトを実行したい時 | `uv run python app.py`             |
| `uv add`   | パッケージを追加し、`pyproject.toml` とロックファイルを更新。       | 新しい依存をプロジェクトに追加したいとき         | `uv add requests`                  |

### Python3.9で作成してください。（バージョンはいずれ上げる予定）

## ./srcフォルダについて
### Modules
- 自作モジュール（importで読み込むやつ）を格納するフォルダ
- 単体で実行した場合も、デモコードが実行される

- noise2noise.py
    - Noise2Noise(Noise_to_Noise)を実装したコード
    - U-Netを使用している
    - [U-Netとは | スキルアップAI Journal](https://www.skillupai.com/blog/tech/segmentation2/)
- ellipse.py
    - アフィン変換（回転、拡大縮小、平行移動）によって楕円補正を試みたやつ
- generate_movie.py
  - 画像の入ったフォルダから、動画を生成するコード
  - 星グル写真のタイムラプスをイメージするとわかりやすいかも
- hsv_low_high.ipynb
  - hsvによる木星の模様検出を試みた、jupyter notebook用のファイル
  - 普通に開くととてつもなく見にくいため、Google Colab上で、閲覧すること推奨
- hsv.py
  - 上のファイルをモジュール化したもの
  - なぜか両方とも結果が違う
- sift.py
  - 特徴点抽出、特徴点マッチングを利用した楕円補正
  - ホモグラフィー変換を使用するため、特徴点マッチングがうまくいかないと失敗する
  - hsv.pyと組み合わせることを想定
  - 簡単なノイズ除去をするクラスも同梱
- ML/affine_image_correction_r1.py
  - 強化学習による、楕円補正の実装（開発中）
  - 2つの画像のシャープネスを測定し、値がいいほうにもう片方を補正する
- ML/image_sharpness.py
  - 使用してないはず（AIに丸々実装させたからわからん）

##＃ mainフォルダについて
- これらのモジュールを使用して、実際にコードを実装したコードを格納するフォルダ

- movie_denoise.py
  - Modules/noise2noise.pyを使用し、フォルダ内のすべての画像ファイルをノイズ除去する
  - ノイズ除去された連番画像から動画ファイルが生成される
- movie_affine.py
  - Modules/ellipse.pyを使用し、フォルダ内のすべての画像ファイルにアフィン変換を適用する





---
このプロジェクトは [MIT License](LICENSE) のもとで公開されています。
また、本プロジェクトで利用しているサードパーティ製ライブラリやパッケージのライセンス情報は、リポジトリ内の「licenses/THIRD_PARTY」ディレクトリに記載されています。各ライブラリのライセンス詳細については、そちらをご参照ください。
