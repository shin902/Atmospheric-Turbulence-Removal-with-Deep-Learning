### 前提

以下を実行

```bash
.venv/bin/activate
```

---

### 画像1枚を処理する

引数一覧
--model: モデルファイルのパス（.pth）
--input: ノイズ除去する画像
--output: ノイズ除去された画像を保存するパス

```bash
python src/Modules/noise2noise.py denoise   --model "Resources/AI/model_dir/fixed_model.pth"   --input "input.jpg" --output "denoised.jpg"
```

---

### 学習を実行する

引数一覧
--train-dir: 学習する画像が入ってるフォルダ
--valid-dir: 検証する画像が入ってるフォルダ
--model-dir: モデルを保存するフォルダ
--epocs: ループの回数（何回同じ画像を連続で学習するか）
--max-pairs: 最大何枚の画像を学習するか
--run-name: モデルファイル、損失などが書かれてるCSVファイルのファイル名

```bash
python src/Modules/noise2noise.py train --train-dir "Resources/AI/train_data" --valid-dir "Resources/AI/valid_data" --model-dir "Resources/AI/model_dir" --epochs 10 --max-pairs 200 --run-name my_model
```


実行すると `--model-dir` に2つのファイルが保存されます。

- `my_model.pth` — 検証ロスが最も良かった時点のモデル（ノイズ除去にそのまま使えます）
- `my_model.csv` — エポックごとの学習・検証ロスの記録

---

### 中断した学習を再開する

```bash
python src/Modules/noise2noise.py train   --train-dir "Resources/AI/train_data"   --valid-dir "Resources/AI/valid_data"   --model-dir "Resources/AI/model_dir"   --epochs 10   --run-name my_model   --resume my_model.pth
```

---

### 対話モードで実行する

引数を覚えていなくても、質問に答えるだけで `train` / `denoise` を実行できます。

```bash
python src/Modules/noise2noise.py interactive
```

引数なしで実行した場合も対話モードになります。

```bash
python src/Modules/noise2noise.py
```

`train` か `denoise` を選んだあと、各項目を順番に入力します。
何も入力せず Enter を押すと `[ ]` 内のデフォルト値が使われます（`--device` や `--resume` は空欄でスキップ可）。


