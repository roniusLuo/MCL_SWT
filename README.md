# Mirror contrastive loss based sliding window transformer (MCL-SWT). 

Written by Jing Luo from Xi'an University of Technology, China.

Jing Luo, Qi Mao, Weiwei Shi, Zhenghao Shi, Xiaofan Wang, Xiaofeng Lu, Xinhong Hei. Mirror contrastive loss based sliding window transformer for subject-independent motor imagery based EEG signal recognition[C]//Human Brain and Artificial Intelligence: Fourth International Workshop, HBAI 2024, Held in Conjunction with IJCAI 2024, Jeju, South Korea, Aug 3-9, 2024. Accepted

E-mail: luojing@xaut.edu.cn

The Sliding Window Transformer model is implemented in SlidingWinTransformer.py. The model's training and testing procedures are conducted in tran_2a.py. Additionally, the utility functions are provided in utils.py to support various operations within the model.

BCI Competition IV dataset 2a and 2b are applied to verify the performance.

The codebase is tested on the following setting.

Python 3.8.8

PyTorch 2.01+cu118

Braindecode 0.4.7

Numpy 1.20.1

Tqdm 4.59.0

Einops 0.6.1

Pandas 1.2.4


