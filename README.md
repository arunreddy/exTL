# extl

Transfer learning refers to the problem of learning and transferring knowledge from a source domain to a new target domain. It has been successfully applied to solve a wide variety of real-world problems. However, due to the difference in data distribution between the source and target domains, existing explanation techniques cannot be readily applied to interpret the transfer learning models. In other words, the key challenge here is how to provide the explanation for the predictive model in the target domain using the information obtained from the source domain (e.g., the influence of examples, relevance of features).

To address this situation, in this research work, we propose an **ex**plainable **T**ransfer **L**earning framework (**exTL**) that learns importance weights of the source domain examples with respect to the model in the target domain, identifies the relevant set of features in the feature space associated with the source domain and provides model explanations in terms of importance weights and relevant features. In our approach, we sample a  mini-batch of examples from the source and target domains, perturb the source domain examples by a small weight and estimate its influence through performance on a set of labeled examples in the target domain. The key idea here is that source domain examples that are similar to those in the target domain need to be up-weighted. We propose an algorithm along with the analysis of the convergence and upper bound on the target domain risk. We demonstrate the effectiveness of the proposed algorithm through the analysis of both text and image data sets. We also explain the transfer learning models through simulated experiments and visualizations. 

> source code for **exTL** paper is hosted in this repository.

- [Installation](#installation)
- [Hardware requirements](#hardware-requirements)
- [Dataset](#dataset)
- [Experiments](#experiments)
  - [Training](#training)




## Installation

The following are the requirements to run exTL framework.
1. Python 3.7
1. PyTorch 1.2 with GPU support
1. Tensorflow 1.4 with GPU support
1. A good NVidia Graphics card.

Install the required packages using `requirements.txt`. Virtual environment is suggested.
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset


## Experiments
To run the experiments, refer the python package `extl.experiments`

