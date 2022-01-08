# References

:books: Paper: [2019 - Generative Adversarial User Model for Reinforcement Learning Based Recommendation System](http://proceedings.mlr.press/v97/chen19f/chen19f.pdf)

:wrench: Repository: [GenerativeAdversarialUserModel](https://github.com/xinshi-chen/GenerativeAdversarialUserModel))

:robot: Algorithm name: GAN-LSTM



# Reproducibility
According to the README file in the original repository

1. Create and activate the Anaconda environemnt 
```bash
➜ conda env create -f rs_survey_python2.yml
➜ conda activate rs_survey_python2
```

2. Download the datasets from the [dropbox](https://www.dropbox.com/sh/57gqb1c98gxasr8/AABDPPVnggypWwn2NsLNq7x6a?dl=0) folder provided by the authors

3. After downloading the .txt files in the shared folder, put then under the 'dropbox' folder, so that the default bash script can automatically find them

4. Process the data
```bash
➜ cd dropbox
➜ ./process_data.sh
```

5. Run the scripts file:
```bash
➜ cd ganrl/experiment_user_model/
➜ ./run_gan_user_model.sh
```
