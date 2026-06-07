# CLIP zero-shot image classification reproduction

本專案依照課堂表格中選定的 5 篇論文進行評估後，選擇第 1 篇論文作為主重現：

**Learning Transferable Visual Models From Natural Language Supervision (CLIP, ICML 2021)**

老師要求「至少選一篇重現」，因此本專案不是額外找新論文，而是從表格中的 5 篇候選論文中選擇 **CLIP** 進行可執行重現。選擇原因是：CLIP 有公開程式碼與預訓練模型，且可以用 CIFAR-100 / CIFAR-10 做 zero-shot 測試，比 Swin、ConvNeXt、nnU-Net、TransUNet 更容易在一般電腦上完成可被檢查的重現流程。

---

## 1. 五篇候選論文與重現選擇

| 排序 | 論文 | 重現建議 |
|---|---|---|
| 1 | CLIP | 主重現：zero-shot image classification |
| 2 | ConvNeXt | 備用：可做 CNN baseline 或 pretrained inference |
| 3 | Swin Transformer | 備用：完整訓練成本高，適合小規模推論/微調 |
| 4 | nnU-Net | 不建議主重現：醫學分割資料與訓練流程較複雜 |
| 5 | TransUNet | 不建議主重現：資料前處理與 GPU 成本較高 |

---

## 2. 重現目標

本專案重現 CLIP 的核心流程：

1. 將影像類別名稱轉成文字提示，例如 `a photo of a cat.`
2. 使用 CLIP text encoder 將文字提示轉成文字特徵。
3. 使用 CLIP image encoder 將影像轉成影像特徵。
4. 計算影像特徵與文字特徵的 cosine similarity。
5. 取相似度最高的文字類別作為預測結果。
6. 輸出 Top-1 Accuracy、Top-5 Accuracy、prediction CSV、confusion matrix CSV。

本專案不重新訓練 4 億組影像文字資料，因為原論文的大規模預訓練成本過高，不適合作為課堂重現。這個限制會在報告 Discussion 中當作差距分析重點。

---

## 3. 建議環境

- Windows 10/11
- Python 3.10
- CPU 可先跑小樣本；有 NVIDIA GPU 會更快
- 建議使用虛擬環境

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

若你有 NVIDIA GPU，也可以依照 PyTorch 官網安裝 CUDA 版本的 torch。

---

## 4. 快速測試：CIFAR-100 小樣本

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

## 5. 完整 CIFAR-100 測試

```powershell
python run_clip_zeroshot.py --dataset cifar100 --model ViT-B/32 --batch-size 64 --max-samples 0
```

`--max-samples 0` 代表跑完整測試集。

---

## 6. 跨資料集測試

老師要求跨資料集時，建議先用 CIFAR-10，因為它公開、下載方便、類別數較少，適合和 CIFAR-100 比較。

```powershell
python run_clip_zeroshot.py --dataset cifar10 --model ViT-B/32 --batch-size 64 --max-samples 0
```

也可以改跑 STL10：

```powershell
python run_clip_zeroshot.py --dataset stl10 --model ViT-B/32 --batch-size 64 --max-samples 0
```

---

## 7. 結果要怎麼填回 Excel

| Excel 欄位 | 填寫方式 |
|---|---|
| 我們重現結果 | 填 CIFAR-100 的 Top-1、Top-5、樣本數、模型名稱 |
| 差距 | 說明這不是完整 ImageNet 重現，不能直接和 ImageNet 76.2% 完全等同 |
| 差距原因分析 | 硬體限制、資料集不同、沒有重新 pretrain、prompt template 不同、樣本數限制 |
| 改善方向 | 增加樣本數、比較多種 prompt、換 ViT-B/16 或 ViT-L/14、加入錯誤類別分析 |
| 跨資料集結果 | 填 CIFAR-10 或 STL10 的 Top-1 / Top-5，並分析不同資料集的泛化差異 |

---

## 8. 建議報告寫法

本研究採低成本重現策略，使用官方 CLIP 預訓練模型重現 zero-shot image classification 流程。由於原論文使用 4 億筆影像文字配對資料進行大規模預訓練，本研究不重新訓練模型，而是以 CIFAR-100 作為可重現測試資料集，驗證文字提示與影像特徵相似度計算是否能完成分類。實驗結果若與原論文有差距，主要原因包含資料集不同、硬體限制、未重新預訓練、prompt 設計差異與測試樣本數限制。除此之外，本研究加入 CIFAR-10 或 STL10 作為跨資料集測試，以觀察 CLIP 在不同資料集上的泛化能力。
