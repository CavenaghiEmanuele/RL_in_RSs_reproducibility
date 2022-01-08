# References

:books: Paper: [2020 - Off-policy Learning in Two-stage Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/3366423.3380130)

:wrench: Repository: [Off-Policy-2-Stage](https://github.com/jiaqima/Off-Policy-2-Stage)

:robot: Algorithm name: 2-Stage


# Reproducibility
According to the README file in the original repository

1. Create and activate the Anaconda environemnt 
```bash
➜ conda env create -f rs_survey.yml
➜ conda activate rs_survey
```

2. Download the [Movielens-1M](https://grouplens.org/datasets/movielens/1m/) dataset
   
3. Unzip and copy it into the 2Stage directory, renaming the directory "movielens"

4. Run the main file:
```bash
➜ python run.py --loss_type loss_ce
➜ python run.py --loss_type loss_ips
➜ python run.py --loss_type loss_2s
```
