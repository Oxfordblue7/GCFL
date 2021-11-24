# Federated Graph Classification over Non-IID Graphs

This repository contains the implementation of the paper:

> [Federated Graph Classification over Non-IID Graphs]()

## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Run repetitions for all datasets
To get all repetition results:

```
bash runnerfile
```
To averagely aggregate all repetitions:

```
python aggregateResults.py --inbase {in_base_path} --outbase {out_base_path} --seq_length {the length of gradient norm sequence}
```

Or, to run one file to get all repetitions also aggregate the results:

```
bash runner.sh
```

### Outputs
The repetition results started with '{\d}_' will be stored in _./outputs/oneDS-nonOverlap-{numClients}clients/{data}/repeats/_ or _./outputs/multiDS-nonOverlap/{data}/repeats/_. 

After aggregating, the two final files are ___avg_accuracy_allAlgos.csv___ and ___stats_performanceGain.csv___ in _./outputs/oneDS-nonOverlap-{numClients}clients/{data}/_ or _./outputs/multiDS-nonOverlap/{data}/_.


## Run once for one certain setting

(1) Distributing one dataset to a number of clients:

```
python main_oneDS.py --repeat {index of the repeat} --data_group {dataset} --num_clients {num of clients} --seed {random seed}  --epsilon1 {epsilon_1} --epsilon2 {epsilon_2} --seq_length {the length of gradient norm sequence}
```

(2) For multiple datasets, each client owns one dataset (datagroups are pre-defined in ___setupGC.py___):

```
python main_multiDS.py --repeat {index of the repeat} --data_group {datagroup} --seed {random seed} --epsilon1 {epsilon_1} --epsilon2 {epsilon_2} --seq_length {the length of gradient norm sequence}
```

*Note: There are various arguments can be defined for different settings. If the arguments 'data_path' and 'outbase' are not specified, datasets will be downloaded in './data' and outputs will be stored in './outputs' by default.

## Acknowledgement
Some codes adapted from [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://github.com/felisat/clustered-federated-learning)
