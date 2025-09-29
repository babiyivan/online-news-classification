# Narrative classification in online news using NLP

Multilingual narrative-classification project for the Natural Language Processing and Information Extraction course, centred on extracting propaganda narratives from online news. Baselines span traditional ML (MultinomialNB, RandomForest) and transformer models (BERT, RoBERTa) evaluated with macro-F1 and related metrics. Deliverables include milestone notebooks, management summary, and final presentation summarising insights and next steps.

For a simple explanation, refer to the [management summary](docs/management-summary.pdf).

This project uses Python 3.12 and [Poetry](https://python-poetry.org/) for dependency management.

## Repository structure
- `notebooks/milestone1.ipynb`: data curation, preprocessing pipeline, and exploratory analysis.
- `notebooks/milestone2_baseline_{bert,roberta}.ipynb`: transformer baselines and evaluation reports.
- `docs/management-summary.pdf`: short overview for stakeholders; `docs/presentation.pdf`: final presentation.

## Setup Instructions

1. **Install Python 3.12** (if not already installed) using [pyenv](https://github.com/pyenv/pyenv):
   ```bash
   pyenv install 3.12
   pyenv local 3.12
   ```

2. **Install Poetry**:
   ```bash
   pip install pipx
   python -m pipx ensurepath
   pipx install poetry
   ```

3. **Add New Dependencies**:
   To add a new library, run:
   ```bash
   poetry add <library>
   ```

4. **Install Project Dependencies**:
   To install all required dependencies, run:
   ```bash
   poetry env use 3.12
   poetry install --no-root
   ```

5. **Install Jupyter** (if not already installed):
   If Jupyter is not installed, you can add it with:
   ```bash
   poetry add jupyter
   ```

6. **Activate the Poetry Virtual Environment**:
   To start working in the virtual environment, type:
   ```bash
   poetry shell
   ```

7. **Run the Jupyter Notebook**:
   Navigate to the notebooks directory and start Jupyter:
   ```bash
   cd notebooks
   jupyter notebook
   ```

Now you're all set to work on the project!

## Milestone 1 Report
All of the code, analysis, and evaluations done as part of milestone 1 can be found [here](/notebooks/milestone1.ipynb)

## Milestone 2 Report
The focus of this report was to implement multiple baseline solutions for our text classification task 

**Extraction of Narratives from Online News - Narrative Classification** and evaluate their performance. This report is split into two parts. The first part will be more in-depth and focus on Traditional Machine learning methods and their performance on this dataset. The second part will describe two deep learning baselines and evaluate their performance.

### Traditional machine learning methods
In this part, we will evaluate the performance of Random Forest, which usually achieves SOTA performance on Tabular datasets and a [Multinomial Naive Bayes Classifier](https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html) which are suitable for classification with discrete features such as word counts. 

For the training data, we use the data files we prepared in Milestone 1 and we encode them using The bag of words method with a feature size of 1024. 

The training and test data have been split in a ratio of 8:2 with a random seed. The training and test are the same for deep learning and traditional ML methods.

Since traditional ML methods are normally faster and easier to train we have built 4 models for each of the currently available languages (two for sub narratives and two for narratives).

For the quantitative analysis, we have decided to focus on the macro f1, precision, and recall score.

An overview of the model performance can be seen below:

| language   | level        | model_name             |   f1_macro |   recal_macro |   precision_macro |
|:-----------|:-------------|:-----------------------|-----------:|--------------:|------------------:|
| PT         | SUBNARATIVES | MultinomialNB          | 0.122312   |    0.212674   |         0.0927896 |
| PT         | SUBNARATIVES | RandomForestClassifier | 0.0203125  |    0.015377   |         0.0416667 |
| PT         | NARATIVES    | MultinomialNB          | 0.325244   |    0.574495   |         0.237612  |
| PT         | NARATIVES    | RandomForestClassifier | 0.143697   |    0.116208   |         0.210498  |
| HI         | SUBNARATIVES | MultinomialNB          | 0.055754   |    0.0589038  |         0.0673611 |
| HI         | SUBNARATIVES | RandomForestClassifier | 0.00677083 |    0.00409226 |         0.0208333 |
| HI         | NARATIVES    | MultinomialNB          | 0.284848   |    0.275325   |         0.338636  |
| HI         | NARATIVES    | RandomForestClassifier | 0.0598485  |    0.0422078  |         0.121212  |
| EN         | SUBNARATIVES | MultinomialNB          | 0.0806928  |    0.122447   |         0.0911334 |
| EN         | SUBNARATIVES | RandomForestClassifier | 0.0167824  |    0.0136846  |         0.026864  |
| EN         | NARATIVES    | MultinomialNB          | 0.281192   |    0.468887   |         0.255638  |
| EN         | NARATIVES    | RandomForestClassifier | 0.0569264  |    0.0475814  |         0.121212  |
| BG         | SUBNARATIVES | MultinomialNB          | 0.0971549  |    0.171875   |         0.0771104 |
| BG         | SUBNARATIVES | RandomForestClassifier | 0          |    0          |         0         |
| BG         | NARATIVES    | MultinomialNB          | 0.274399   |    0.447452   |         0.218578  |
| BG         | NARATIVES    | RandomForestClassifier | 0.052381   |    0.0368687  |         0.113636  |

This data can be further aggregated to get a better understanding of the model's performance.

#### Aggregation by model

|Model|   f1_macro mean |   f1_macro std  |  recal_macro mean  |  recal_macro std  |   precision_macro mean |  precision_macro std |
|-|-----------------------:|----------------------:|--------------------------:|-------------------------:|------------------------------:|-----------------------------:|
|MultinomialNB|              0.1902    |             0.110775  |                 0.291507  |                0.184909  |                     0.172357  |                    0.10282   |
|RandomForrest|              0.0445899 |             0.0463668 |                 0.0345024 |                0.0374664 |                     0.0819903 |                    0.0714341 |

If we satisfy the data by model, we can see that surprisingly the best-performing model is the Multinomial Naive Bayes and not the RandomForrest model, which normally achieves SOTA performance on tabular data

#### Aggregation by language

|Language|   f1_macro mean|   f1_macro std |   recal_macro mean |   recal_macro std |   precision_macro mean |   precision_macro', 'std |
|-|-----------------------:|----------------------:|--------------------------:|-------------------------:|------------------------------:|-----------------------------:|
|PT|               0.152891 |              0.126889 |                 0.229688  |                 0.243576 |                      0.145642 |                    0.0935754 |
|EN|               0.108898 |              0.117852 |                 0.16315   |                 0.208828 |                      0.123712 |                    0.096353  |
|BG|               0.105984 |              0.11909  |                 0.164049  |                 0.202867 |                      0.102331 |                    0.0908275 |
|HI|               0.101805 |              0.124388 |                 0.0951321 |                 0.122299 |                      0.137011 |                    0.140536  |

If we aggregate the data by language, we notice some unexpected results. Before the analysis, we assumed that English would have the best results since English is the most widely supported language in NLP, but it seems that the accuracy for the Portuguese data is much better. It could be because the PT dataset is more balanced, or maybe it contains a lot more 'Other' labels which are normally the most prevalent in the dataset and easier to classify.

#### Which Narrative labels are the easiest to predict
To score used to calcučate how good label predictions are is obtained by the following formula:
$$
score = \frac{1}{n}\sum_{i=0}^{n}abs(I - p)
$$
where $I$ is an indicator varia equal to 1 or 0 and $p$ is the prediction probability of the model.



| label                                                  |   mean probability error |   Number of training datapoints |   Number of test datapoints |
|:-------------------------------------------------------|-------------------------:|--------------------------------:|----------------------------:|
| Other                                                  |                 0.712814 |                              80 |                          17 |
| URW: Blaming the war on others rather than the invader |                 0.765873 |                              15 |                           3 |
| CC: Criticism of climate policies                      |                 0.783331 |                               7 |                           3 |
| URW: Discrediting the West, Diplomacy                  |                 0.785071 |                              24 |                          11 |
| URW: Speculating war outcomes                          |                 0.818252 |                              11 |                           4 |
| CC: Criticism of institutions and authorities          |                 0.824094 |                              16 |                           3 |
| URW: Russia is the Victim                              |                 0.834108 |                               7 |                           5 |
| URW: Amplifying war-related fears                      |                 0.849082 |                              21 |                           5 |
| URW: Overpraising the West                             |                 0.849702 |                               8 |                           1 |
| CC: Controversy about green technologies               |                 0.857633 |                               4 |                           1 |
| CC: Criticism of climate movement                      |                 0.859328 |                              11 |                           1 |
| CC: Hidden plots by secret schemes of powerful groups  |                 0.864364 |                               6 |                           0 |
| URW: Discrediting Ukraine                              |                 0.873203 |                              13 |                           6 |
| URW: Praise of Russia                                  |                 0.89781  |                               7 |                           3 |
| CC: Questioning the measurements and science           |                 0.916215 |                               3 |                           1 |
| CC: Downplaying climate change                         |                 0.924982 |                               1 |                           1 |
| URW: Negative Consequences for the West                |                 0.936809 |                               6 |                           1 |
| CC: Green policies are geopolitical instruments        |                 0.946143 |                               1 |                           0 |
| URW: Hidden plots by secret schemes of powerful groups |                 0.949995 |                               6 |                           2 |
| URW: Distrust towards Media                            |                 0.974954 |                               7 |                           2 |
| CC: Climate change is beneficial                       |                 0.975    |                               0 |                           1 |
| CC: Amplifying Climate Fears                           |                 1        |                               0 |                           0 |

Based on the results we can see that we struggle with the prediction of Narrative Labels that have a small amount of training examples. Based on the data it seems like the prediction score is only related to how many training data points we have.

All of the code related to the quantitative analysis, building of models, and the required data transformations can be found [here](/notebooks/milestone2-traditional_ml_methods.ipynb)

#### Qualitative analysis
In reviewing the mismatches found across the 5 samples, several key observations and potential patterns were identified.

##### Inconsistent classification for similar statements
A recurring mismatch involves statements that discuss similar topics or sentiments but are classified differently by the model. For example, in sample 1 "CC: Criticism of climate movement: Ad hominem attacks on key activists"** and "CC: Downplaying climate change: Ice is not melting" are classified as True in the ground truth, yet predicted as False. These errors may point to misclassification in cases where there are nuanced debates around climate change policies or activism. These topics might overlap with broader political and social ideologies, which makes the classification task challenging.

##### Ideological complexity
Several labels in the mismatch list involve ideological positions that can be interpreted in multiple ways depending on one’s perspective. For example, I sample the statement "URW: Amplifying war-related fears: The West will attack other countries" is difficult because it involves predicting a future political scenario with implications for geopolitics. The label itself is a reflection of a specific ideological viewpoint (the belief that the West might provoke a larger conflict). The model may have trouble distinguishing between valid concerns and propagandist narratives.

##### "Other" label
In sample1 and sample3, many mismatches occur for statements labeled "Other". For example, the statement "CC: Criticism of climate policies: Other" is misclassified as False in the sample. The label "Other" generally indicates broad or less easily definable categories, or at least the label that applies in case none of the other labels in the same topic (in this case "Criticism of climate policies") apply. This could also point to a dataset challenge where labels within the “Other” category are either too vague or inconsistently applied, making it harder for the model to develop a reliable classification rule for such instances.

##### Ambiguity
The model seems to struggle when faced with complex or ambiguous phrasing. For instance, in sample 2, the statement "CC: Criticism of climate policies: Climate policies have a negative impact on the economy" was misclassified, indicating that the model may be challenged by nuanced criticisms that require a deeper understanding of the economic implications of climate policies.

Similarly, sample 4 contains multiple predictions around political and war-related fears (e.g., "The West will attack other countries" and "The real possibility of nuclear weapons"), where the model might be overlooking subtle differences in the phrasing or context in these complex political arguments.

##### Conclusion
Several key takeaways from the analyzed mismatches (quantitative analysis) are:

More often than not, the model incorrectly predicts 'False' when the actual label is 'True', this seems to be a pattern across the sampled cases, the only exception being sample1.

The model may struggle because the statements involve subjective interpretations that depend on political ideology, which may cause it to misclassify them.

Labels such as "Other" and other generalized criticisms (e.g., "Climate policies are ineffective") seem to be frequently misclassified. The model might fail to grasp the broader context or interpret these vague categories correctly, leading to errors.

Statements that critique complex subjects like climate change policies or specific persons (e.g., "Ad hominem attacks on key activists") involve subtle language and multi-layered arguments, which might be difficult for the model to properly interpret, causing misclassification.

The full qualitative analysis can be found [here](/notebooks/milestone2-traditional-ml-evaluation.ipynb)


### Deep learning ROBERTA


[The classification report](/notebooks/milestone2_roberta.ipynb) and the metrics provided indicate that the model's performance is quite poor. For most classes, the precision, recall, and F1-score are 0.00, indicating that the model failed to correctly classify any instances for these categories. This is a significant issue as it shows the model is not learning to distinguish between different classes effectively. The number of instances for each class is very low, which could be contributing to the poor performance. Classes with only a few instances are particularly challenging for the model to learn.

The overall metrics are as follows: The micro average precision is 0.45, recall is 0.11, and F1-score is 0.17. These values suggest that while the model is somewhat precise when it does make a prediction, it misses a large number of true positives, leading to a low recall and F1-score. The macro average precision is 0.00, recall is 0.01, and F1-score is 0.01. These extremely low values indicate that the model performs poorly across all classes, without favoring any particular class. The weighted average precision is 0.05, recall is 0.11, and F1-score is 0.07. These values are slightly better than the macro average but still indicate poor overall performance, considering class imbalance. The sample's average precision is 0.45, recall is 0.23, and F1-score is 0.30. These values are somewhat better, suggesting that the model performs slightly better when considering individual samples rather than aggregated class performance.

Additional metrics include a Hamming Loss of 0.0467, which indicates the fraction of labels that are incorrectly predicted. A lower value is better, so this is relatively low, but given the other metrics, it doesn't compensate for the poor performance. The Macro F1 is 0.0067, which is extremely low, reinforcing the poor performance across all classes. The Micro F1 is 0.1731, which is slightly better but still indicates poor performance. The Subset Accuracy is 0.0, indicating that the model did not perfectly predict any instance, which is a critical issue.

A quick overview of the Roberta model performance can be found in the table below:

|              | precision | recall | f1   | support |
|--------------|-----------|--------|------|---------|
| micro avg    | 0.45      | 0.11   | 0.17 | 168     |
| macro avg    | 0.00      | 0.01   | 0.01 | 168     |
| weighted avg | 0.05      | 0.11   | 0.07 | 168     |
| samples avg  | 0.45      | 0.23   | 0.30 | 168     |

### Deep learning BERT

[The classification report for BERT](/notebooks/milestone2_baseline_bert.ipynb)

Similarly to RoBERTa, The classification report and metrics indicate that the model struggles to correctly identify most of the classes. The precision, recall, and F1-score for the majority of categories are 0.00, which may stem from class imbalance and very low support for many categories, making it difficult for the model to learn useful patterns.

The overall macro metrics are particularly telling: a Macro F1 of 0.02 and macro-level precision and recall near zero indicate that the model is not performing well on any class consistently. While the micro and sample averages are a bit higher, the improvement is still minimal. The low subset accuracy and the low F1 scores confirm that the model is not providing reliable multi-label predictions.

These results suggest the need for a more balanced dataset, more extensive training, or possibly a different model architecture or input representation to improve performance. 

A quick overview of the baseline BERT performance for the English dataset is found below:

|              | precision | recall | f1   | support |
|--------------|-----------|--------|------|---------|
| micro avg    | 0.11      | 0.31   | 0.16 | 300     |
| macro avg    | 0.01      | 0.09   | 0.02 | 300     |
| weighted avg | 0.12      | 0.31   | 0.17 | 300     |
| samples avg  | 0.11      | 0.52   | 0.17 | 300     |

### Comparison of baseline models
The table below contains the performance metrics of all the baseline models:

|                       | macro f1 | macro precision | macro recall |
|-----------------------|----------|-----------------|--------------|
| RandomForest Baseline | 0.016    | 0.026           | 0.013        |
| MultinomialNB         | 0.08     | 0.09            | 0.122        |
| BERT                  | 0.02     | 0.01            | 0.09         |
| Roberta               | 0.01     | 0.00            | 0.01         |

Even tho BERT and ROBERTA are state-of-the-art Models pre-trained on large datasets, they are still not able to learn enough from the small dataset with sparse labels to beat simple, but more robust models Such as the MultinomialNb, which uses word counts as a feature. This is a good starting point for our future models, as we know now that simple text features such as word counts should be enough to reach a macro F1 score  ~0.08


### Division of work
- Tibor Cus - building traditional ML models, quantitative analysis, report
- Ahmed Sabanovic - ROBERTA deep learning baseline, qualitative and quantitative analysis
- Theo Hauray - BERT deep learning baseline, qualitative and quantitative analysis
- Ivan Babiy - Traditional ML methods Qualitative analysis