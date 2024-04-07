# DBTKT

*Modeling Learning Transfer Effects in Knowledge Tracing: A Dynamic and Bidirectional Perspective* (accepted at DASFAA'2024)


### Implementation

Pytorch

### Usage

You need create folders `output/, logs/, data/processed/` and put your downloaded data in `data/`. You can download all raw dataset (ASSITS2012/ASSIST2017/NIPS2020) through [Edudata](https://github.com/bigdata-ustc/EduData), such as

```
edudata download assist2017
```



#### Preprocess

```
./preprocess.sh
```

or run `preprocess.py`. Then, preprocessed data will be in `data/processed/`.

#### Train & eval

```
./run.sh
```

or run `run.py` through command line.
