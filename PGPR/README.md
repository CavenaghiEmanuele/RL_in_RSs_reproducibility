# References

:books: Paper: [2019 - Reinforcement Knowledge Graph Reasoning for Explainable Recommendation](https://arxiv.org/abs/1906.05237)


:wrench: Repository: [PGPR](https://github.com/orcax/PGPR)

:robot: Algorithm name: PGRP


# Reproducibility
According to the README file in the original repository

1. Create and activate the Anaconda environemnt 
```bash
➜ conda env create -f rs_survey.yml
➜ conda activate rs_survey
```

2. Download the datasets from [here](https://drive.google.com/uc?export=download&confirm=Tiux&id=1CL4Pjumj9d7fUDQb1_leIMOot73kVxKB)

3. Copy the datasets into a "data" directorty in the PGPR folder


4. Proprocess the data first (<dataset_name> should be one of "cd", "beauty", "cloth", "cell" (refer to utils.py)):
```bash
python preprocess.py --dataset <dataset_name>
```

5. Train knowledge graph embeddings (TransE in this case):
```bash
python train_transe_model.py --dataset <dataset_name>
```

6. Train RL agent:
```bash
python train_agent.py --dataset <dataset_name>
```

7. Evaluation
```bash
python test_agent.py --dataset <dataset_name> --run_path True --run_eval True
```
