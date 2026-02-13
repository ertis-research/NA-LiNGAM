This repository has all the code used for testing the novel algorithm NA-LiNGAM. It has several scripts to test the different aspects of the algorithm, including execution time, accuracy identifying the non-noise variables and the accuracy generating the causal DAG.

In the following points the functioning of the code is explained:

## NA-LiNGAM algorithm

The code related to the NA-LiNGAM algorithm itself is composed by two scripts: *nalingam_environment* and *nalingam_score*, both in the subfolder *utils*. To use it, it is only neccesary to use *GraphEnvNALiNGAM*, which contains all the funtions to use the algorithm. The explanation about the functioning of the environment and its methods are located in the script.

## Datasets generators

There are two types of dataset generators: synthetic and real data generators. Both of them have the methods neccesary to obtain all the data and graph information about them after the generation process. The real data generator has implemented the [Sachs et Al., 2005](https://www.science.org/doi/10.1126/science.1105809) dataset about cellular signaling networks using protein measurements from human cells of the immune system, and a use case about consumption in our research building (however this dataset is private). Also the synthetic generator has a function called *generate_samples*, which generates a new set of nodes and relationships to the already set parameters.

## Test scripts

In the main folder there are several *test_* scripts that can be used in order to test the different aspects of the graph composition process and obtain its metrics.