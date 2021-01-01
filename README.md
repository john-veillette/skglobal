# a scikit-learn style classifier for optimizing global objectives

This repository contains some linear classifiers that can optimize specific "global objectives," such as ROC-AUC or precision-at-recall, directly using (optionally stochastic) gradient descent. This is in contrast to most classifiers in the scikit-learn arsenal that optimize some differentiable surrogate for accuracy, even if they'll ultimately be evaluated using a different, global objective function. 

Our estimators conform to the [scikit-learn](https://scikit-learn.org/stable/) estimator API and extend the [MNE-Python](https://mne.tools/stable/index.html) `LinearModel` class. It may be desirable to remove the dependency on MNE for general use, but the intended application here is multivariate-pattern analysis of EEG data so MNE integration is handy.

## Loss Functions

The loss functions used come from a [no-longer-supported Google research repo](https://github.com/tensorflow/models/tree/archive/research/global_objectives). Since their code is archived, we copied it into the `global_objectives` directory with an `__init__.py` file added for easier access while maintaining a clear deliniation between their work and mine. I also replaced some `tf.contrib` functionality and modified the import statements for Tensorflow 2 compatibility. If you use this repository, make sure to cite their gem of a [paper](https://arxiv.org/abs/1608.04802).