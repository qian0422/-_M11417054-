# CLIP zero-shot image classification reproduction


**Learning Transferable Visual Models From Natural Language Supervision (CLIP, ICML 2021)**


---

## 1. 實作目標

本專案重現 CLIP 的核心流程：

1. 將影像類別名稱轉成文字提示，例如 `a photo of a cat.`
2. 使用 CLIP text encoder 將文字提示轉成文字特徵。
3. 使用 CLIP image encoder 將影像轉成影像特徵。
4. 計算影像特徵與文字特徵的 cosine similarity。
5. 取相似度最高的文字類別作為預測結果。
6. 輸出 Top-1 Accuracy、Top-5 Accuracy、prediction CSV、confusion matrix CSV。

本專案不重新訓練 4 億組影像文字資料，因為原論文的大規模預訓練成本過高，不適合作為課堂重現。這個限制會在報告 Discussion 中當作差距分析重點。

---


```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```


---

## 2. 快速測試：CIFAR-100 小樣本

```powershell
python run_clip_zeroshot.py --dataset cifar100 --model ViT-B/32 --batch-size 64 --max-samples 1000
```

輸出會在：

```text
results/metrics.csv
results/predictions_cifar100_ViT-B-32_basic.csv
results/confusion_matrix_cifar100_ViT-B-32_basic.csv
```

---

## 3. 完整 CIFAR-100 測試

```powershell
python run_clip_zeroshot.py --dataset cifar100 --model ViT-B/32 --batch-size 64 --max-samples 0
```

`--max-samples 0` 代表跑完整測試集。

---

## 4. 跨資料集測試

建議先用 CIFAR-10，因為它公開、下載方便、類別數較少，適合和 CIFAR-100 比較。

```powershell
python run_clip_zeroshot.py --dataset cifar10 --model ViT-B/32 --batch-size 64 --max-samples 0
```

也可以改跑 STL10：

```powershell
python run_clip_zeroshot.py --dataset stl10 --model ViT-B/32 --batch-size 64 --max-samples 0
```

---
