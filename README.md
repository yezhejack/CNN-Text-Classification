# CNN-Text-Classification
Implementing CNN for Movie Review accoding to [Kim (2014)](https://arxiv.org/abs/1408.5882).

## Prepare data
```
python process_data.py <path_to_word2vec>
```

## Start Training
### static
```
python main.py --embedding_freeze
```
Acc=0.812
### non-static
```
python main.py
```
Acc=0.806
## Dependencies ##
* pytorch==0.3.0
* python3
* numpy