#   AudioDetector

## AED

声音场景分类，Audio Event Detection (AED)，有很多方法：

* MFCC 等特征 + GMM / HMM / NMF / SVM 等分类器
* log-mel spectrogram + CNN

论文 [2] 发现视频分类任务中，简单地把单帧图片分类结果求平均方法，效果和复杂模型一样好。

受 [2] 启发， [1] 基于声音片段分类结果求平均的方法，横向对比了多种基于 CNN 的 AED 模型。

## 训练经验

MFCC 特征比 Mel-spectrogram 训练效果更好。训练集正负样本各 900 个，用七层 CNN 训练模型后，用测试集测试，MFCC 特征的 EER 0.28 AUC 0.82，Mel-spectrogram 的 EER 0.44 AUC 0.55。

样本稀缺，尝试 data augmentation / data synthesis


## Setup

pip3 install librosa playsound pyobjc sklearn

## Dataset

https://research.google.com/audioset/balanced_train/laughter.html

## Reference

[1] Hershey, Shawn, et al. "CNN architectures for large-scale audio classification." Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. IEEE, 2017.

[2] Ng, Joe Yue-Hei, et al. "Beyond short snippets: Deep networks for video classification." Computer Vision and Pattern Recognition (CVPR), 2015 IEEE Conference on. IEEE, 2015.

[3] Knox, Mary Tai, and Nikki Mirghafori. "Automatic laughter detection using neural networks." Eighth Annual Conference of the International Speech Communication Association. 2007. [PDF](http://www.icsi.berkeley.edu/pubs/speech/laughter_v10.pdf)