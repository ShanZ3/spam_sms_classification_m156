### Spam SMS Classification
> Data Source:<br>
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection <br>

> Goal:<br>
Apply learned machine learning algorithms to classify SMS message (Spam or not Spam) <br>

> Procedure:
> + Use TF-IDF to extract 30-50 features (number of unique words) over SMS corpus and create weights of the features on bag-of words
> + Use CountVectorizer to create token counts as sparse matrix
> + Classification: Navie Bayes, SVM, Random Forest
> + Deep Learning: Neural Nets
> + Use GridSearchCV to find out the optimum hyperparameters (stratified cv)
> + Model Performance Evaluation: select the best performing model.

