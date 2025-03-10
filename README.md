# News Lens

## Dataset

[Article-Bias-Prediction](https://github.com/ramybaly/Article-Bias-Prediction) dataset curated by Baly et al. (2020) is used in this project. The dataset contains over 30,000 news articles labeled with bias labels: left, center, and right. The dataset is split into training, validation, and test sets.

```
@inproceedings{baly2020we,
  author      = {Baly, Ramy and Da San Martino, Giovanni and Glass, James and Nakov, Preslav},
  title       = {We Can Detect Your Bias: Predicting the Political Ideology of News Articles},
  booktitle   = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  series      = {EMNLP~'20},
  NOmonth     = {November},
  year        = {2020}
  pages       = {4982--4991},
  NOpublisher = {Association for Computational Linguistics}
}
```

## Models

1. Logistic Regression
2. Random Forest
3. LlaMA: See the [README](llama/README.md) for more details

## File Structure

```
llama: contains the LlaMA model and the evaluation of the model
    ├── README.md
    ├── llama_train.py: QLoRA finetuning script for the LlaMA model
    ├── llama_distill.py: Script to further finetune the LlaMA model on DeepSeek labels
    ├── Llama.ipynb
    ├── match_articles.py: Script to detect test data leakage
Random_forest.ipynb: contains the random forest model and the evaluation of the model
RQ_EDA_Preprocessing_logistic_stcking_Evaluation.ipynb: contains the EDA, preprocessing, logistic regression, and stacking models
Sentence_transformer_visualization.ipynb: contains the preliminary exploration and visualization with sentence transformer models including BGE-m3
requirements.txt: contains the required libraries to run the code
```
