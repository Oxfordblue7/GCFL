# Federated Graph Classification over Non-IID Graphs

This repository contains the implementation of the paper:

> [Federated Graph Classification over Non-IID Graphs](https://arxiv.org/pdf/2106.13423.pdf)

## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Run once for one certain setting

(1) OneDS: Distributing one dataset to a number of clients:

```
python main_oneDS.py --repeat {index of the repeat} --data_group {dataset} --num_clients {num of clients} --seed {random seed}  --epsilon1 {epsilon_1} --epsilon2 {epsilon_2} --seq_length {the length of gradient norm sequence}
```

(2) MultiDS: For multiple datasets, each client owns one dataset (datagroups are pre-defined in ___setupGC.py___):

```
python main_multiDS.py --repeat {index of the repeat} --data_group {datagroup} --seed {random seed} --epsilon1 {epsilon_1} --epsilon2 {epsilon_2} --seq_length {the length of gradient norm sequence}
```


## Run repetitions for all datasets
(1) To get all repetition results:

```
bash runnerfile
```
(2) To averagely aggregate all repetitions, and get the overall performance:

```
python aggregateResults.py --inpath {the path to repetitions} --outpath {the path to outputs} --data_partition {the data partition mechanism}
```

Or, to run one file for all:

```
bash runnerfile_aggregateResults
```

### Outputs
The repetition results started with '{\d}_' will be stored in:
> _./outputs/seqLen{seq_length}/oneDS-nonOverlap/{dataset}-{numClients}clients/eps\_{epsilon1}\_{epsilon2}/repeats/_, for the OneDS setting;

> _./outputs/seqLen{seq_length}/multiDS-nonOverlap/{datagroup}/eps\_{epsilon1}\_{epsilon2}/repeats/_, for the MultiDS setting.

After aggregating, the two final files are: 
> ___avg_accuracy_allAlgos_GC.csv___, which includes the average performance over clients for all algorithms;

> ___stats_performanceGain_GC.csv___, which shows the performance gain among all clients for all algorithms.


*Note: There are various arguments can be defined for different settings. If the arguments 'datapath' and 'outbase' are not specified, datasets will be downloaded in './data', and outputs will be stored in './outputs' by default.

## Acknowledgement
Some codes adapted from [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://github.com/felisat/clustered-federated-learning)
