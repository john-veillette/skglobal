# a scikit-learn style classifier for optimizing global objectives

This repository contains some linear classifiers that can optimize specific "global objectives," such as ROC-AUC or precision-at-recall, directly using (optionally stochastic) gradient descent. This is in contrast to most classifiers in the scikit-learn arsenal that optimize some differentiable surrogate for accuracy, even if they'll ultimately be evaluated using a different, global objective function. 

Our estimators conform to the [scikit-learn](https://scikit-learn.org/stable/) estimator API and extend the [MNE-Python](https://mne.tools/stable/index.html) `LinearModel` class. It may be desirable to remove the dependency on MNE for general use, but the intended application here is multivariate-pattern analysis of EEG data so MNE integration is handy.

## How to Use

In brief, you can import the estimator, initialize it with `LinearClassifier(loss, **loss_args)`, and then use it just like any other (binary) linear classifier in the scikit-learn library. The available loss functions are currently `'precision_recall_auc'`, `'roc_auc'`, `'recall_at_precision'`, `'precision_at_recall'`, `'false_positive_rate_at_true_positive_rate'`, and `'true_positive_rate_at_false_positive_rate'`. Check `global_objectives/loss_layers.py` to see what `**loss_args` arguments your chosen loss function takes.

You can also specify a regularization `penalty` and weight `C`, `tol`, and `max_iter`, which work consistently with how they would in an sklearn classifier. If you want to use stochastic gradient descent for optimization, just specify `batch_size`.

The only argument the `fit` method can take other than `X` and `y` is `sample_weight`, but I'm not actually sure that works with the global loss functions yet. You should probably ignore it for now. 

Some super minimal example code is in `example.ipynb`.

__Note__: Since the loss functions are "global" and not dependant on a particular decision threshold, the classifier doesn't really optimize the intercept of the linear model. So you may have to enforce your own decision threshold at inference time, depending on your usage case.

## Loss Functions

The loss functions used come from a [no-longer-supported Google research repo](https://github.com/tensorflow/models/tree/archive/research/global_objectives). Since their code is archived, we copied it into the `global_objectives` directory with an `__init__.py` file added for easier access while maintaining a clear deliniation between their work and mine. I also replaced some `tf.contrib` functionality and modified the import statements for Tensorflow 2 compatibility. If you use this repository, make sure to cite their gem of a [paper](https://arxiv.org/abs/1608.04802).
