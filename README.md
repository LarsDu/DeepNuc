# DeepNuc
DeepNuc is an implementation of Convolutional Neural Networks (CNN) in Tensorflow for making binary classification of nucleotide sequences. For example, you can build a classifier for transcriptional start sites (TSS) of a genome traing it with transcriptional start site sequences as the positive class, and dinucleotide shuffled sequence as the negative class. DeepNuc also include grid search and k-folds cross validation for hyperparameter optimization.

Check the [manual](DeepNuc_Instructions.pdf) for more information.

As part of my research, I looked into applying Deep Taylor Decomposition to build a picture of where imporant features being used for making inferences are located within sequences. This technique for decomposing models trained on images actually is not particularly informative for DNA sequences, but the layer classes I built to achieve this are available under deepnuc/dtlayers.py. The classes in this file can be used to perform Deep Taylor Decomposition on images, and the demo examples for Deep Taylor Decomposition are located under demos.
