# SGEM: Test-Time Adaptation for Automatic Speech Recognition via Sequential-Level Generalized Entropy Minimization
![](res/concept_figure.png)

### Abstract
Given a CTC-based trained ASR model, we proposed **Single-Utterance Test-time Adaptation (SUTA)** framework, which can adapt the source ASR model for one utterance by unsupervised objectives (such as entropy minimization, minimum class confusion). For details of SUTA's method and experimental results, please see our paper [[link]](https://arxiv.org/abs/2203.14222).

Our proposed SUTA has below advantages:
* Efficient adaptation for one **single utterance**
* Don't need to access the source data
* About 0.1s per 1s utterance for 10-steps adaptation

### Environmental Setup 
```
pip install -r requirements.txt
```

### Data Preparation
Currently, our code only supports [Librispeech](https://www.openslr.org/12)/[CHiME-3](https://catalog.ldc.upenn.edu/LDC2017S24)/[Common voice En](https://tinyurl.com/cvjune2020)/[TED-LIUM 3](https://www.openslr.org/51/)
You have to download datasets by your own.

### Run
The source ASR model is [w2v2-base fine-tuned on Librispeech 960 hours](https://huggingface.co/facebook/wav2vec2-base-960h). The pre-trained model is imported by Huggingface.

Run SUTA on different datasets:
```
bash scripts/{dataset_name: LS/CH/CV/TD}.sh
```

### Contact 
* Changhun Kim changhun.kim@kaist.ac.kr

### Citation
```
@inproceedings{kim2023sgem,
  title={{SGEM}: Test-Time Adaptation for Automatic Speech Recognition via Sequential-Level Generalized Entropy Minimization},
  author={Changhun Kim and Joonhyung Park and Hajin Shim, and Eunho Yang},
  booktitle={Conference of the International Speech Communication Association (INTERSPEECH)},
  year={2023}
}
```
