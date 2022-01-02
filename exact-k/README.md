# Paper
2020 [Exact-K Recommendation via Maximal Clique Optimization](https://doi.org/10.1145/3292500.3330832)

# Algorithm name
IRecGAN

# Reproducibility
According to the original README file in the paper repository:

1. Create and activate the Anaconda environemnt 
```bash
➜ conda env create -f rs_survey_python2.yml
➜ conda activate rs_survey_python2
```

2. Create the data directory and load the CIKM Cup 2016 dataset from [here](https://cloud.tsinghua.edu.cn/f/e4bb57a633074009a1eb/) into the data directory.
```bash
➜ cd src
➜ mkdir data
```

3. To run the code on the CIKM CUP 2016 dataset run:
```bash
➜ python main.py
```
