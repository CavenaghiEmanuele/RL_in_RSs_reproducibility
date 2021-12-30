# Paper
2019 [Reinforced Negative Sampling over Knowledge Graph for Recommendation](https://dl.acm.org/doi/abs/10.1145/3366423.3380098)

# Algorithm name
KGPolicy

# Reproducibility
According to the README file reported in the original paper's repository:

1. Get dataset and pretrain model (the dataset and the pretrained model are not included in the repository)
```bash
➜ wget https://github.com/xiangwang1223/kgpolicy/releases/download/v1.0/Data.zip
➜ unzip Data.zip
```

2. Create and activate the Anaconda environemnt 
```bash
➜ conda create -n rs_survey_kgpolicy python=3.6
➜ conda activate rs_survey_kgpolicy
```
3. Install the missing pip dependencies:
```bash
pip install prettytable
```

4. Switch to source code dir 
```bash
➜ cd kgpolicy
```

5. Ensure python version is `3.6`. Then install all requirements for this project.
```bash
➜ bash setup.sh
```

6. Train KG-Policy on:

- `last-fm` dataset. 
```bash
➜ python main.py
```

- `amazon-book` dataset
```bash
➜ python main.py --regs 1e-4 --dataset amazon-book --model_path model/best_ab.ckpt 
```

- `yelp2018` dataset
```bash
➜ python main.py --regs 1e-4 --dataset yelp2018 --model_path model/best_yelp.ckpt 
```
