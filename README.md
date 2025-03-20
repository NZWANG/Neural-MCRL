<div align="center">

<h1 style="border-bottom: 1px solid lightgray;">Neural-MCRL: Neural Multimodal Contrastive Representation Learning for EEG-based Visual Decoding</h2>

<!-- Badges and Links Section -->
<div style="display: flex; align-items: center; justify-content: center;">

<p align="center">
  <a href="#">
  <p align="center">
    <a href='https://arxiv.org/abs/2412.17337'>
  </p>
</p>


</div>

<br/>

</div>

<img src="Neural-MCRL.png" alt="Neural-MCRL" style="max-width: 80%; height: auto;"/>
Overall framework of the Neural-MCRL.

<!-- ## Usage -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Usage</h2>
This repo is the official implementation of Neural-MCRL: Neural Multimodal Contrastive Representation Learning for EEG-based Visual Decoding


<!-- ## Data availability -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Data availability</h2>
You can download the relevant THINGS-EEG data set and THINGS-MEG data set at osf.io.
The raw and preprocessed EEG dataset, the training and test images are available on [osf](https://osf.io/3jk45/).

<!-- ## EEG preprocessing -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">EEG preprocessing</h2>

Modify your path and execute the following code to perform the same preprocessing on the raw data as in our experiment:
```
cd EEG-preprocessing/
python EEG-preprocessing/preprocessing.py
```
Also You can get the data set used in this project through the BaiduNetDisk [link](https://pan.baidu.com/s/1-1hgpoi4nereLVqE4ylE_g?pwd=nid5) to run the code.

<!-- ## Acknowledge -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Acknowledge</h2>

1. Song Y, Liu B, Li X, et al. Decoding natural images from eeg for object recognition[J]. arXiv preprint arXiv:2308.13234, 2023.
2. Li D, Wei C, Li S, et al. Visual decoding and reconstruction via eeg embeddings with guided diffusion[J]. arXiv preprint arXiv:2403.07721, 2024.
3. Gifford A T, Dwivedi K, Roig G, et al. A large and rich EEG dataset for modeling human visual object recognition[J]. NeuroImage, 2022, 264: 119754.
4. Grootswagers T, Zhou I, Robinson A K, et al. Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams[J]. Scientific Data, 2022, 9(1): 3.


<!-- ## Citation -->
<h2 style="border-bottom: 1px solid lightgray; margin-bottom: 5px;">Citation</h2>

```bibtex
@article{li2024neural,
  title={Neural-MCRL: Neural Multimodal Contrastive Representation Learning for EEG-based Visual Decoding},
  author={Li, Yueyang and Kang, Zijian and Gong, Shengyu and Dong, Wenhao and Zeng, Weiming and Yan, Hongjie and Siok, Wai Ting and Wang, Nizhuan},
  journal={arXiv preprint arXiv:2412.17337},
  year={2024}
}
```
