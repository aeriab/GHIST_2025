**This repository was used by Brendan Aeria to compete in the 2025 Genomic History Inference Strategies Tournament with the Garud Lab.**

The strategy being implemented involves a Domain Adversarial Neural Network (DANN) which trains using SLiM simulations while adapting to the domain of GHIST competition data. To be specific, this model uses a discriminator branch which is connected to the same last layer of the convolutional neural network that is used for classification of "neutral" or "sweep".


The "TrainingDANN" folder contains the training scripts I used to train the binary classifier DANN. This also used GHIST 2025 chromosome 21 data as target data.

The "ProcessingData" folder is how I obtained the simulated neutral and sweep .npy files.

The "ReformatGHISTData" contains scripts that were used to convert the GHIST .tsv file into a .npy file that can be read directly into the DANN for training.

The "Old" folder contains failed attempts to convert SLiM simulations into usable .npy arrays for training. There are also a number of other failed attempts that were sent to this folder.
