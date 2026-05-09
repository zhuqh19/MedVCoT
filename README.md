# MedVCoT: Bridging Modality Gap in Medical VQA through Latent Visual Reasoning

<p align="center">
  <a href="">📄 Paper</a> •
  <a href="https://zhuqh19.github.io/MedVCoT/">🌐 Project Page</a>
</p>

---

## 📖 Introduction

We propose MedVCoT, the first approach to incorporate a Visual Chain-of-Thought (Visual CoT) into medical VQA. Different from traditional methods, which only perform reasoning in the text space, our model generates a sequence of continuous latent visual tokens autonomously within a separate thought horizon (denoted as \texttt{<think>} tags) before answering. These tokens act as a bridge between linguistic reasoning and visual perception. In particular, under the guidance of the specialized medical visual expert MedSAM, the VLM learns to generate these tokens as an intermediate reasoning output. These generated tokens can be further mapped with a projector to trigger MedSAM to decode explicit segmentation masks. This mechanism forces the model to “see” and localize the lesion in latent space before “speaking” the diagnosis so as to ensure that answers are causally predicated on verifiable visual evidence.

---

## 📊 Performance

### Main Result
<p align="center">
  <img src="./statics/images/main_result.png" width="34%">
</p>


### Abalation study
<p align="center">
  <img src="./statics/images/ablation.png" width="90%">
</p>

### Fine-grained capability analysis
<p align="center">
  <img src="./statics/images/radar.png" width="45%">
</p>

### Case study
<p align="center">
  <img src="./statics/images/case_study.png" width="45%">
</p>

---

## 📊 Dataset

- [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)
- [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa)
- [VQA-Med-2019](https://github.com/abachaa/VQA-Med-2019)

---


## 📚 Citation

If you find this work useful, please cite:

```bibtex
to be released
```

---

## ⭐ Acknowledgement

- [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [CoVT](https://github.com/Wakals/CoVT)
- [Lingshu](https://alibaba-damo-academy.github.io/lingshu/)
