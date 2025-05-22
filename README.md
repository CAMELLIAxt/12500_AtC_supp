# README

This repository contains the code and data for the Aggregate-then-Calibrate (AtC) framework, a two-stage method that combines human judgments with predictive models to produce calibrated scores. In AtC, stage 1 aggregates multiple annotators’ pairwise comparisons into a consensus ranking (modeling each annotator’s reliability), and stage 2 calibrates a model’s raw scores so that they respect the human-derived ranking (using isotonic regression). By combining the ordinal information (from humans) and the scoring scale (from the model), AtC produces final scores that align with human consensus while leveraging model consistency. You can also check the online [anonymous git repo](https://anonymous.4open.science/r/23845_AtC_supp-838E/) .

## Datasets

We include two semi-synthetic datasets used in our experiments. In each case the “ground truth” scores are known, and we simulate the imperfect model prediction:

- Reading Level: This dataset contains pairwise comparisons of text documents based on reading difficulty. The dataset includes 490 documents with known reading levels, serving as ground truth difficulty scores. A total of 624 annotators provided 12728 pairwise judgments, with each comparison indicating which document is easier to read. The dataset structure includes judge information, judgment outcomes (where "A" indicates document A is easier than document B), document identification numbers, and the corresponding reading levels for each document pair.

- Country Population: This dataset uses the populations of the 15 countries (true populations in millions are the ground truth). This dataset consists non-expert annotators who compare countries by population (e.g. “Does country X have a larger population than country Y?”), with noise (e.g. confusing similarly sized populations or bias toward familiar countries). The task is to aggregate these 105 pairwise judgments into a ranking of countries and estimate each country’s population. An “oracle” model score s_p is created by adding noise to the true populations.

The initial model outputs (s_p) are synthetic by design (true score plus noise). All required data files are provided under the Datas/ directory (Country_pop_datas and Reading_level_datas_reg10), including pairwise comparison lists and ground-truth scores.

## Reproducing Experiments

The scripts for the semi-synthetic experiments corresponding to Research Questions 1–3 are provided in the Codes/ directory. The results for RQ1–3 can be reproduced by running the following steps (no additional setup needed):
### Reading Level experiment:
Run the Python script for the reading level dataset:
```
python Codes/Reading_level.py
```
This script uses annotator comparisons on the Reading Level data as input, aggregates rankings (Stage 1), trains a simple model to predict passage difficulty with noise, and applies the AtC calibration (Stage 2). It outputs evaluation metrics (e.g. Kendall’s $\tau$, Wasserstein distance, KS statistic, MSE) comparing the aggregated score, raw model score, and calibrated score against the true readability scores.

<!-- ### Country Population experiment:
Run the Python script for the country population dataset:
```
python Codes/Country_pop.py
```
This script performs the analogous procedure for the Country Population data. It simulates pairwise population comparisons, applies rank aggregation, trains a noisy predictor for population, and then calibrates via AtC. It computes distributional metrics (KL divergence, JS divergence) and other measures for the aggregate score, model score, and calibrated score against the true populations. -->

### Analysis of isotonic calibration:
Optionally, run the analysis script for isotonic behavior:
```
python Codes/analyze_isotonic_behavior.py
```
This produces additional plots and data (e.g. demonstrating how AtC handles intentional ranking errors with annotator-specific noise added to simulate heterogeneity) related to RQ3. The core isotonic regression implementation is in Codes/isotonic.py.

After running these scripts, you should obtain results comparable to those reported for RQ1–RQ3 (see our paper’s Table 1 and related figures). The scripts print the evaluation metrics for the uncalibrated human aggregate ($s^*$), the raw model predictions ($s_p$), and the final AtC-calibrated scores ($\hat{s}$) on each dataset.

## Code Organization
-	Codes/ – contains the implementation. The main analysis scripts are at the top level (Reading_level.py, Country_pop.py, analyze_isotonic_behavior.py, isotonic.py). The Codes/HRAs/ subdirectory contains our implementation of the heterogeneous rank aggregation (HRA) model used in Stage 1 (with example scripts work_readinglevel.m and work_countrypopulation_final_HRAs.m).
- Datas/ – contains data files for each dataset.
- Datas/Country_pop_datas/ includes the country population ground truth (Country_pop_doc_info.txt), pairwise comparisons (Country_pop_all_pair.txt), annotator data (Country_pop_HRA_*.xlsx), etc.
- Datas/Reading_level_datas_reg10/ includes the reading passages info (Reading_level_doc_info.txt), pairwise comparisons (Reading_level_all_pair.txt), and related files (reg10_score_Ones_+_HRA-*_s->a.txt).
