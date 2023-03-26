# GCN for Link Prediction
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3109/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

This project explores the use of Graph Convolutional Networks (GCNs) for link prediction in graphs. The GCN model is implemented using the Deep Graph Library (DGL) in Python.

## Requirements

- Python 3
- PyTorch
- DGL
- NumPy

## Usage

The main code for the GCN model is implemented in `link_predict.py`. To train and test the model on ogbl-colab, run the following command:
```
python link_predict.py
```
The trained model will be saved in the root directory, and the training loss and hit@50 scores will be printed to the console. The script supports the following command-line arguments:
- --save: type=int, default=10, help='specify how many epochs to run before saving the model'
- --epochs: type=int, default=10, help='specify how many epochs to run in total'
- --batch_size: type=int, default=4096, help='specify the batch size during training and testing'
- --eval_steps: type=int, default=100, help='specify how many steps to run before evaluating the model'
''' 


## Dataset

The ogbl-colab dataset from Open Graph Benchmark [1] is used in this project. It consists of a heterogeneous graph with 3 types of nodes (papers, authors, and institutions) and 4 types of edges (author-paper, paper-author, paper-institution, and institution-paper).

## Acknowledgements

This project was inspired by the GCN implementation in the DGL documentation [2]. The ogbl-colab dataset is from the Open Graph Benchmark [1]. Special thanks to ChatGPT for providing guidance and support throughout the project.

## References

[1] Open Graph Benchmark (ogb). https://ogb.stanford.edu/

[2] Deep Graph Library (DGL) Documentation. https://docs.dgl.ai/en/latest/index.html
