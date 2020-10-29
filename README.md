# MONSTOR: An Inductive Approach for Estimating and Maximizing Influence over Unseen Networks

Source code for MONSTOR, described in the paper [MONSTOR: An Inductive Approach for Estimating and Maximizing Influence over Unseen Networks](https://arxiv.org/pdf/2001.08853.pdf).

MONSTOR (**Mon**te Carlo **S**imula**tor**) is an inductive machine learning method for estimating the influecne of given seed nodes in social networks unseen during training. To the best of our knowledge, MONSTOR is the first inductive method for this purpose. MONSTOR can greatly accelerate existing IM algorithms by replacing repeated MC simulations.

## Requirements

To install requirements, run the following command on your terminal:
```setup
pip install -r requirements.txt
```

Also, since some files (datadir/*.pkl.gz, graphs/scal_*.txt) exceed GitHub's file size limit, please download the files via [Dropbox Link](https://bit.ly/3e9UnLb).

## Generating training data

To generate the training data for the model, run these commands:

```
./compile.sh
./monte_carlo_[random|degree] graphs/[Extended|Celebrity|WannaCry]_[train|test]_[BT|JI|LP].txt
python processing.py [Extended|Celebrity|WannaCry]_[train|test]_[BT|JI|LP]
```
You should create a directory (e.g., Extended_train_BT ) in raw_data before you run the commands.
Since we provide preprocessed data, you don't have to run these commands to train and evaluate the model.

## Training

To train the model in the paper, run this command:

```train
python train.py --target=[Extended|Celebrity|WannaCry] --input-dim=4 --hidden-dim=16 --gpu=0 --layer-num=3 --lamb=0.3 --epochs=100
```
```
arguments:
  -h, --help            show this help message and exit
  --target TARGET       choose the target graph for masking
  --input-dim INPUT_DIM
                        input dimension
  --hidden-dim HIDDEN_DIM
                        hidden dimension
  --gpu GPU             gpu number
  --layer-num LAYER_NUM
                        number of layers
  --lamb LAMB           hyperparameter lambda (in Equation (4))
  --epochs EPOCHS       number of epochs
```

> ex) if target is set to 'Extended', then the model will be trained with 'Celebrity' and 'WannaCry'.

You can see pretrained models in example_checkpoints folder. (The format of file names is [Extended|Celebrity|WannaCry]_[BT|JI|LP][1-6].pkt)

## Evaluation

* IE.py: Experiment regarding Influence Estimation
* IM.py: Experiment regarding Influence Maximization
* submodularity.py: Experiment regarding Empirical submodularity of MONSTOR
* scalability.py: Experiment regarding Empirical scalability of MONSTOR

To evaluate the model, run:

### For IE / IM / submodularity.py
```eval
python [IE|IM|submodularity].py --input-dim=4 --hidden-dim=16 --layer-num=3 --gpu=0 --checkpoint-path=[path_of_target_checkpoint] --prob=[BT|JI|LP] --n-stacks=[number_of_stacks]
```
```
arguments:
  -h, --help            show this help message and exit
  --input-dim INPUT_DIM
                        input dimension
  --hidden-dim HIDDEN_DIM
                        hidden dimension
  --layer-num LAYER_NUM
                        number of layers
  --checkpoint-path CHECKPOINT_PATH
                        path of the target checkpoint
  --prob PROB           target activation probablity
  --n-stacks N_STACKS   number of stacks
  --gpu GPU             gpu number
```

### For scalability.py
```eval
python scalability.py --graph-path=graphs/scal_[20|21|22|23|24|25|26].txt --input-dim=4 --hidden-dim=16 --gpu=0 --layer-num=3 --checkpoint-path=[path_of_target_checkpoint]
```
```
arguments:
  -h, --help            show this help message and exit
  --graph-path GRAPH_PATH
                        path of the graph for evaluation
  --input-dim INPUT_DIM
                        input dimension
  --hidden-dim HIDDEN_DIM
                        hidden dimension
  --layer-num LAYER_NUM
                        number of layers
  --checkpoint-path CHECKPOINT_PATH
                        path of the target checkpoint
  --gpu GPU             gpu number
```

### For computing the average influence of MC simulations
To compute the average influence of 100,000 MC simulations, run:
```eval
./compile.sh
./test [path_of_target_graph] [path_of_seeds]
```
> ex) ./test graphs/Extended_test_BT.txt example_seeds.txt

## Terms and Conditions

If you use this code as part of any published research, please consider acknowledging our paper.

```
@inproceedings{ko2020monstor,
  title={MONSTOR: An Inductive Approach for Estimating and Maximizing Influence over Unseen Networks},
  author={Ko, Jihoon and Lee, Kyuhan and Shin, Kijung and Park, Noseong},
  booktitle={IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)},
  year={2020},
}
```
