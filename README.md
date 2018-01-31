#   AudioDetector

## AED

声音场景分类，Audio Event Detection (AED)，有很多方法：

* MFCC 等特征 + GMM / HMM / NMF / SVM 等分类器
* log-mel spectrogram + CNN



论文 [2] 发现视频分类任务中，简单地把单帧图片分类结果求平均方法，效果和复杂模型一样好。

受 [2] 启发， [1] 基于声音片段分类结果求平均的方法，横向对比了多种基于 CNN 的 AED 模型。



## Setup

pip3 install librosa playsound pyobjc

## Reference

[1] Hershey, Shawn, et al. "CNN architectures for large-scale audio classification." Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on. IEEE, 2017.

[2] Ng, Joe Yue-Hei, et al. "Beyond short snippets: Deep networks for video classification." Computer Vision and Pattern Recognition (CVPR), 2015 IEEE Conference on. IEEE, 2015.