## Background
Walmart is a renowned retail corporation that operates a chain of hypermarkets. Here, Walmart has provided a data combining of 45 stores including store information and monthly sales. The data is provided on weekly basis. Walmart tries to find the impact of holidays on the sales of store. For which it has included four holidays’ weeks into the dataset which are Christmas, Thanksgiving, Super bowl, Labor Day. Here we are owing to Analyze the dataset given. Before doing that, let me point out the objective of this analysis.

## Business Objectives
Our Main Objective is to predict sales of store in a week. As in dataset size and time related data are given as feature, so analyze if sales are impacted by time-based factors and space- based factor. Most importantly how inclusion of holidays in a week soars the sales in store?

![Feature Description]('T:\SalePrediction\data\sale_data\BigMartSales Prediction\Description.png')


## Machine Learning Algorithm Used
**RandomForestRegressor**

We will use the sklearn module for training our random forest regression model, specifically the RandomForestRegressor function. The RandomForestRegressor documentation shows many different parameters we can select for our model. Some of the important parameters are highlighted below:

* n_estimators — the number of decision trees you will be running in the model
criterion — this variable allows you to select the criterion (loss function) used to determine model outcomes. We can select from loss functions such as mean squared error (MSE) and mean absolute error (MAE). The default value is MSE.

* max_depth — this sets the maximum possible depth of each tree
max_features — the maximum number of features the model will consider when determining a split

* bootstrap — the default value for this is True, meaning the model follows bootstrapping principles (defined earlier)
max_samples — This parameter assumes bootstrapping is set to True, if not, this parameter doesn’t apply. In the case of True, this value sets the largest size of each sample for each tree.

* Other important parameters are min_samples_split, min_samples_leaf, n_jobs, and others that can be read in the sklearn’s RandomForestRegressor documentation [here 💁‍♂️](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)