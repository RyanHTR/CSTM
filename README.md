# CSTM

Implementation for Collaborative Spatial-Temporal Modeling for Language-Queried Video Actor Segmentation, CVPR 2021. [PDF](https://arxiv.org/abs/2105.06818).

## Setup
* Python3.8
* PyTorch 1.9
* CUDA 11

Download A2D Sentences dataset from [this link](https://kgavrilyuk.github.io/publication/actor_action/). Modify the paths of dataset in `./datasets/a2d_dataset.py`. The GloVe word embeddings can be downloaded from [this link](https://nlp.stanford.edu/projects/glove/) and put `glove.840B.300d.zip` in `./word_embedding`.

## Running
We provide the trained checkpoint of our model in [Baidu Drive](https://pan.baidu.com/share/init?surl=ZuQd3-97A-K-B_pVOSTNjQ), password: fs58.

For testing on A2D Sentences dataset, please use the following command:

```
python test.py \
--data_margin 2 --batch_size 8 \
--gpu_id 0 --resize 320 --skip single \
--dim_semantic 512 \
--dataset A2D --model_root cstm_checkpoints \
--checkpoint checkpoint.pth.tar
```