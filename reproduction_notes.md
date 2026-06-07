# Reproduction Notes for Excel and Final Report

## Selected paper
Learning Transferable Visual Models From Natural Language Supervision (CLIP)

## Why this paper was selected
CLIP is suitable for course reproduction because the official pretrained model can be used directly for zero-shot image classification. This avoids the unrealistic requirement of retraining on 400M image-text pairs, while still reproducing the core mechanism of the paper.

## Core method to reproduce
1. Convert class names into text prompts.
2. Encode images and prompts using CLIP.
3. Compute cosine similarity between image and text features.
4. Choose the text class with the highest similarity.
5. Report Top-1 and Top-5 accuracy.

## Gap analysis draft
The reproduced result is expected to differ from the paper result because the original CLIP paper reports large-scale results on datasets such as ImageNet, while this project uses CIFAR-100 or another smaller public dataset for practical reproduction. In addition, this project does not retrain CLIP from 400M image-text pairs. Hardware limitations, prompt design, batch size, and dataset domain differences may also affect the final accuracy.

## Proposed improvement draft
Future work can test multiple prompt templates, increase the number of evaluated samples, use a larger CLIP backbone such as ViT-B/16 or ViT-L/14, and compare the model on both object-centric and scene-centric datasets. A confusion matrix and error analysis can also be added to understand which classes are difficult for zero-shot recognition.
