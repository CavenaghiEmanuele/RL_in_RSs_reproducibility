# Paper
2020 [Self-Supervised Reinforcement Learning for Recommender Systems](https://doi.org/10.1145/3397271.3401147)

# Algorithm name
RL4REC

# Reproducibility
There is no documentation in the original document.

1. Create and activate the Anaconda environemnt 
```bash
➜ conda env create -f rs_survey_tensorflow1.yml
➜ conda activate rs_survey_tensorflow1
```

### To run the experiments on Kaggle (RetailRocket) dataset  

1. go to the Kaggle directory:
```bash
➜ cd Kaggle
```

2. Download the dataset from [here](https://drive.google.com/file/d/1nLL3_knhj_RbaP_IepBLkwaT6zNIeD5z/view). Extract the folder and copy the RL4REC/Kaggle/data directory into the directory RL4REC/Kaggle.

3. Run the main file of each algorithm:
```bash
➜ python [ALGORITHM NAME].py
```

### To run the experiments on RC15 dataset  

1. go to the RC15 directory:
```bash
➜ cd RC15
```

2. Download the dataset from [here](https://drive.google.com/file/d/1nLL3_knhj_RbaP_IepBLkwaT6zNIeD5z/view). Extract the folder and copy the RL4REC/RC15/data directory into the directory RL4REC/RC15.

3. Run the main file of each algorithm:
```bash
➜ python [ALGORITHM NAME].py
```
