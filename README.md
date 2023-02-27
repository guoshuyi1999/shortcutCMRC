#  Shortcut Learning on Conversational Machine Reading Comprehension: A Thorough Empirical Study

 

This is the repository for the paper Shortcut Learning on Conversational Machine Reading Comprehension: A Thorough Empirical Study. 


## Get Dataset

To get the original CoQA/QuAC/SQuAD 2.0 dataset and the attacked CoQA/QuAC/SQuAD 2.0 dataset 

 

### Prepare GloVe

```
mkdir glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove/glove.840B.300d.zip
unzip glove/glove.840B.300d.zip -d glove
```


## BERT Experiments

### Preprocess

```
cd BERT/
bash preprocess_QuAC.sh
bash preprocess_CoQA.sh
bash preprocess_SQuAD.sh
```

### Train & Predict & Calculate Score

To train:

```
cd BERT/
bash train_QuAC.sh
bash train_CoQA.sh
bash train_SQuAD.sh
```

To predict:

```
cd BERT/
bash predict-coqa.sh
bash predict-quac.sh
bash predict_squad.sh
```

To calculate score:

```
cd BERT/src/
bash score-coqa.sh
bash score-quac.sh
bash score-squad.sh
```


## FlowQA Experiments

### Prepreocess

```
cd FlowQA/
bash preprocess_QuAC.sh
bash preprocess_CoQA.sh
bash preprocess_SQuAD.sh
```

### Train & Predict & Calculate Score

To train:

```
cd FlowQA/
bash train_QuAC.sh
bash train_CoQA.sh
bash train_SQuAD.sh
```

To predict:

```
cd FlowQA/
bash predict-coqa.sh
bash predict-quac.sh
bash predict_SQuAD.sh
```

## Contact

If you have any questions, please new an issue or contact me,*E-mail: shuyi21@mails.jlu.edu.cn*.