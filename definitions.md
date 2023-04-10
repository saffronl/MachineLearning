Precision - from all classes predicted as positive, how many are actually positive

Recall - from all the positive classes, how many were predicted correctly

Accuracy - from all classes (postive & negative), how many were predicted correctly

Confusion Matrix - A confusion matrix is helpful for measuring recall, precision, specificity, and accuracy. The matrix is a 2x2 with true positive, true negative, false positive, and false negative.

Evaluate the accuracy of the model using the testing dataset. Cross validation will provide three values from the number of "k-folds" specified. Cross-Validation has two main steps: splitting the data into subsets (called folds) and rotating the training and validation among them. https://towardsdatascience.com/what-is-cross-validation-60c01f9d9e75

Specificity - the model's ability to predict a true negative

FPR - false positive rate (1-specificity)

Sensitivity - a measure of how well a model can detect positive instances (inverse of specificity)

ROC Curve - recall, specificity, false positive rate, sensitivity

Bias - is when som aspects of a dataset are given more weigh/representation than others = skewed outcome, low accuracy, errors

Variance - refers to the changes in the model when using different portions of the training dataset.

Bias vs. variance - as a model gets more complex, the bias may decrease but the variacne increases. although the model may fit the training data better, it will fit the testing data less due to low bias but HIGH variance between models.

Cross Fold - In k-fold cross-validation, you split the input data into k subsets of data (also known as folds). You train an ML model on all but one (k-1) of the subsets, and then evaluate the model on the subset that was not used for training.

Regression is when we predict quantitative outputs

Classification is whn we predict qualitative outputs

The linear modl makes huge assumptions about structure and ields stable but possibly inaccurate preditictions. The K-nearest neighbords makes very mild structural assumptions: its precitions are opften accurable but unstable

The B is the intercept, also knows as bias, also known as the pull towards outliers.

The method of least squares chooses the coefficients B to minimive the residual sum of squares.

Statistical decision theory - requires a loss function for penalizing errors in prediction, most commonly using eh squared error loss

bayes classifier - error rate of the bayes classifier is called the bayes rate

Two learning techniques for predictions: stable but biased linear model and the less stable but apparently less biased class of knn esitimate. it would seem that with a reasonably large set of training dat, we could always approvismate the theoretically optimal conditional expection by knn averatging, since we should be able to find a fairly large neighborhood of ovservationsc ose to any x and average them. this approach and our intuition breaks down in high dimemsions, commonly regerred to as the curse of dimensionality. By imposing some heaby restrictions on the class of models being fitted, we can aboid the curse of dimensionality.

bias- variance decomposition = breaking down mean squared error into variance and bias ( mse = var + bias^2 )

the complexity of functions f many ariables can grow exponentially with the dimension, and if we wish to be able to estimate such functions with the same accuracy as functions in low dimsions then we need the size of our training set to grow exponentially as well.

Bias-variance tradeoff - as the model complexity of out procedure is increaased, the variance tends to increase and the squared bias tends to decrease, the opposite behavior occurs as the model completxity is dereased. for knn, the model complexity is controlled by k.

Typically, we would like to choose out model complexity to trade bias off with variance in such a way as to minimize the test errors. an obvcious estimate of test error is the training error. Unfotunately, training error is not a good esitimate of test error.

The training error tends to dcrease whenever we increase the model complatxity (OVERFITTING). However, will too much fitting, the model adapts itself too closely to the training data and will not generalize well (have a large test error). In that case the predictions will have a large variance. In contrast, if the model is not complex enough, it will underfit and may have large bias, again resulting in poor generalization.

The f statistic measures the change in residual sum-of-squares per additional parameter in the bigger model, and it is normalized by an estimate of variance.


Least Squares Estimate - estimateing parameters by minimizing the squared discrepancies between observed data, on the one hand, and their expected values on the other. It is the most widely used procedure for developing estimates of the model paramters.

There are two reasons why we are often not satistied with the least squares estimates: 1. prediction accuracy: the least squares estimates often have low bias but large cariance. prediction accuracy can sometimes be improved by shrinking or seettings ome coefficients to zero. 2. interpretation: with a large number of precictors, we often would like to determine a smaller subset that exhibit the strongest effects. in order to get the "big picure", we are willing to sacrifice some of the small details. 

Forward-stepwise selection starts with the intercept, and then sequentially adds into the model the predictor that most improves the fit.

backwards-stepwise selection starts wiht the full model, and sequentially deletes the predictor that has the least impact on the fit. The candidate for dropping is the variable with the smallest z-score 

Forward-stagewise regression- is more constrained than forward-stepqise regression. it starts like forward-stepwise, but at each step the algo identifies the variable most correlated with the current residual. it then computes the simple linear regression coefficient of the residual onthis chosen variable, and then adds it to the curren coefficient for that variable.













