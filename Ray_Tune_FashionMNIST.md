This notebook can be found on https://github.com/RaphaelMaser/Ray_Tune or opened in colab with the button below

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RaphaelMaser/Ray_Tune/blob/main/Ray_Tune_FashionMNIST.ipynb)

<img src="./images/ray_header_logo.png" 
     align="middle" 
     width="1000" 
     title="Ray Logo (https://docs.ray.io/en/latest/index.html)"/>

*Ray Logo (https://docs.ray.io/en/latest/index.html)*

## Ray

Ray is a framework for distributed computing in Python. The core idea of Ray is to make distributed computing as accessible as possible while not disturbing the workflow of the user. For that purpose it integrates well with all of the mainstream Python libraries used in distributed scenarios like PyTorch, Tensorflow, Scikit-learn, etc.. Integrating Ray in your own workflow typically only needs small changes in your code, there is no need to re-write your application.

In general Ray consists of several different libraries:

<img src="./images/what-is-ray-padded.svg" 
     align="center" 
     width="1200" />

*Ray Framework Overview (https://docs.ray.io/en/latest/index.html)*

Ray Core provides the capability to build clusters and distribute any workload across the nodes of the cluster. Whether that be your PC and laptop or some gpu nodes of the iris-HPC. Ray schedules the tasks across the nodes and manages data transfers between the nodes.

Ray Core is a low-level library and although it is well made it needs some knowledge to use it for more sophisticated tasks like hyperparameter optimization (HPO) or NN training. Ray's higher-level libraries like Ray Train (NN training) and Ray Tune (HPO) use the Ray Core library to provide a simple interface for interacting with Ray without needing to know the details of distributed training.

With Ray Tune only a few lines of code needs to be added to the training procedure to run a full HPO with state-of-the-art search algorithms and schedulers.

### Why should I use Ray Tune and not one of the other HPO libraries?

<img src="./images/tune_overview.webp" 
     align="center" 
     width="600" />

*Ray Framework Overview (https://docs.ray.io/en/latest/index.html)*

Ray Tune itself provides a rich library of state-of-the-art algorithms for HPO. It furthermore integrates well with existing hyperparameter search libraries like optuna or hyperopt. The advantage in using Ray Tune for that purpose is twofold.

Ray Tune and the integrated libraries provide a wide range of algorithms. If you use Tune and want to change between different HPO libraries you don't need to rewrite any code, just change the search algorithm. If you would not use Tune a change between libraries would probably lead to rewriting some of the training code to match the requierements of the new library.

Ray provides a high degree of scalability. The trials are running concurrently on your machine and ressources of each trial can be easily managed (basically a one-liner). If you are training a complex model with huge datasets you can easily scale the workload across several nodes on the HPC. Scaling the workload from one node on the HPC to a whole cluster of nodes on the HPC requires a minimum of effort. 

Therefore if you are not sure whether you want to use the distributed capabilities of Ray or not I would recommend it. Even without the need for distributed computations it provides a great framework for HPO and if you decide that you need to scale the workload it can be done easily. But using the distributed capabilities of Ray really makes sense in most medical imaging situations. Even if you use only one node on the HPC it still provides 4 gpus. If we would do HPO search on this node using standard PyTorch without Ray we could probably use the DistributedDataParallel class to use all 4 gpus for the training of the model. This would probably create overhead because PyTorch needs to synchronize the state of the model across the gpus. Since HPO always needs some useful number of trails (probably higher than 4) we could also use Ray to train for models in parallel on the node to avoid this communication overhead.

## Defining a simple Neural Network for the FashionMNIST Dataset in PyTorch

Here the usage of HPO using Ray Tune is demonstrated on the FashionMNIST dataset with a simple NN. First we need to install the required dependencies and import all needed classes.


```python
!pip install -q ray[tune]==2.2.0
!pip install -q torch==1.13.1 torchvision==0.14.1
!pip install -q matplotlib==3.6.2
```


```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split
from matplotlib import pyplot as plt

from ray import tune, air
from ray.air import session
import ray
import os
from ray.tune.schedulers import ASHAScheduler
from ray.air import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
```

Afterwards we need to download the FashionMNIST dataset and create a PyTorch dataset from that. Luckily PyTorch can automatically download the dataset if it is not found in the directory. Afterwards we create two dataloaders, one for training and one for validation. In this example we split the official train set from FashionMNIST in a two parts for that purpose. The validation set will be used to search for good hyperparameter combinations and validate the model during training. The test dataset from FashionMNIST could be used afterwards to compare the network with the standard parameters and the models optimized with HPO. 


```python
# Download training data from open datasets.
data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Split the train set and create data loaders
train_data, val_data = random_split(data, [0.8,0.2])
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size)
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz


    0.9%

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz


    100.0%


    Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw


    100.0%

    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
    Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw


    
    3.0%

    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz


    100.0%


    Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz


    100.0%

    Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
    


    


The model used for the demonstration is a very simple fully connected network with 3 layers.


```python
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

We use a standard PyTorch training routine for the model. It would also be possible to use PyTorch Ignite or Lightning to avoid boilerplate code or use even other machine learning frameworks like Tensorflow or Keras.


```python
# Standard PyTorch training routine
def train_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct*100

```

The train() function iterates through the epochs and executes the train_epoch() function in every step. Furthermore it uses the test() function to test the model with the data in val_dataloader and show the current accuracy. For the training we use SGD with a learning rate of 1e-4 and a weight decay of 1e-5.


```python
# Small function which iterates through the epochs
def train(epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-04, weight_decay=1e-05)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(train_dataloader, model, loss_fn, optimizer, device)
        test(val_dataloader, model, loss_fn, device)

# Start the training
train(10)
```

    Epoch 1
    -------------------------------
    Test Error: 
     Accuracy: 17.9%, Avg loss: 2.278881 
    
    Epoch 2
    -------------------------------
    Test Error: 
     Accuracy: 21.1%, Avg loss: 2.266609 
    
    Epoch 3
    -------------------------------
    Test Error: 
     Accuracy: 25.1%, Avg loss: 2.254291 
    
    Epoch 4
    -------------------------------
    Test Error: 
     Accuracy: 28.4%, Avg loss: 2.241845 
    
    Epoch 5
    -------------------------------
    Test Error: 
     Accuracy: 30.7%, Avg loss: 2.229174 
    
    Epoch 6
    -------------------------------
    Test Error: 
     Accuracy: 32.6%, Avg loss: 2.216170 
    
    Epoch 7
    -------------------------------
    Test Error: 
     Accuracy: 33.7%, Avg loss: 2.202708 
    
    Epoch 8
    -------------------------------
    Test Error: 
     Accuracy: 35.1%, Avg loss: 2.188691 
    
    Epoch 9
    -------------------------------
    Test Error: 
     Accuracy: 36.5%, Avg loss: 2.174028 
    
    Epoch 10
    -------------------------------
    Test Error: 
     Accuracy: 37.7%, Avg loss: 2.158642 
    


## Hyperparameter optimization (Random Search)

Now we are going to use Ray to do a small random hyperparameter search. 

The following line is not mandatory, normally Ray initializes itself if it was not initialized before, but in this case I want to set the log_to_driver argument to "False" to suppress the output of the trials. Otherwise the output of the different trails will be mixed in no special order because of the concurrent execution.


```python
ray.init(log_to_driver=False, ignore_reinit_error=True)
```

    2023-01-09 15:40:44,466	INFO worker.py:1370 -- Calling ray.init() again after it has already been called.





<div>
    <div style="margin-left: 50px;display: flex;flex-direction: row;align-items: center">
        <h3 style="color: var(--jp-ui-font-color0)">Ray</h3>
        <svg version="1.1" id="ray" width="3em" viewBox="0 0 144.5 144.6" style="margin-left: 3em;margin-right: 3em">
            <g id="layer-1">
                <path fill="#00a2e9" class="st0" d="M97.3,77.2c-3.8-1.1-6.2,0.9-8.3,5.1c-3.5,6.8-9.9,9.9-17.4,9.6S58,88.1,54.8,81.2c-1.4-3-3-4-6.3-4.1
                    c-5.6-0.1-9.9,0.1-13.1,6.4c-3.8,7.6-13.6,10.2-21.8,7.6C5.2,88.4-0.4,80.5,0,71.7c0.1-8.4,5.7-15.8,13.8-18.2
                    c8.4-2.6,17.5,0.7,22.3,8c1.3,1.9,1.3,5.2,3.6,5.6c3.9,0.6,8,0.2,12,0.2c1.8,0,1.9-1.6,2.4-2.8c3.5-7.8,9.7-11.8,18-11.9
                    c8.2-0.1,14.4,3.9,17.8,11.4c1.3,2.8,2.9,3.6,5.7,3.3c1-0.1,2,0.1,3,0c2.8-0.5,6.4,1.7,8.1-2.7s-2.3-5.5-4.1-7.5
                    c-5.1-5.7-10.9-10.8-16.1-16.3C84,38,81.9,37.1,78,38.3C66.7,42,56.2,35.7,53,24.1C50.3,14,57.3,2.8,67.7,0.5
                    C78.4-2,89,4.7,91.5,15.3c0.1,0.3,0.1,0.5,0.2,0.8c0.7,3.4,0.7,6.9-0.8,9.8c-1.7,3.2-0.8,5,1.5,7.2c6.7,6.5,13.3,13,19.8,19.7
                    c1.8,1.8,3,2.1,5.5,1.2c9.1-3.4,17.9-0.6,23.4,7c4.8,6.9,4.6,16.1-0.4,22.9c-5.4,7.2-14.2,9.9-23.1,6.5c-2.3-0.9-3.5-0.6-5.1,1.1
                    c-6.7,6.9-13.6,13.7-20.5,20.4c-1.8,1.8-2.5,3.2-1.4,5.9c3.5,8.7,0.3,18.6-7.7,23.6c-7.9,5-18.2,3.8-24.8-2.9
                    c-6.4-6.4-7.4-16.2-2.5-24.3c4.9-7.8,14.5-11,23.1-7.8c3,1.1,4.7,0.5,6.9-1.7C91.7,98.4,98,92.3,104.2,86c1.6-1.6,4.1-2.7,2.6-6.2
                    c-1.4-3.3-3.8-2.5-6.2-2.6C99.8,77.2,98.9,77.2,97.3,77.2z M72.1,29.7c5.5,0.1,9.9-4.3,10-9.8c0-0.1,0-0.2,0-0.3
                    C81.8,14,77,9.8,71.5,10.2c-5,0.3-9,4.2-9.3,9.2c-0.2,5.5,4,10.1,9.5,10.3C71.8,29.7,72,29.7,72.1,29.7z M72.3,62.3
                    c-5.4-0.1-9.9,4.2-10.1,9.7c0,0.2,0,0.3,0,0.5c0.2,5.4,4.5,9.7,9.9,10c5.1,0.1,9.9-4.7,10.1-9.8c0.2-5.5-4-10-9.5-10.3
                    C72.6,62.3,72.4,62.3,72.3,62.3z M115,72.5c0.1,5.4,4.5,9.7,9.8,9.9c5.6-0.2,10-4.8,10-10.4c-0.2-5.4-4.6-9.7-10-9.7
                    c-5.3-0.1-9.8,4.2-9.9,9.5C115,72.1,115,72.3,115,72.5z M19.5,62.3c-5.4,0.1-9.8,4.4-10,9.8c-0.1,5.1,5.2,10.4,10.2,10.3
                    c5.6-0.2,10-4.9,9.8-10.5c-0.1-5.4-4.5-9.7-9.9-9.6C19.6,62.3,19.5,62.3,19.5,62.3z M71.8,134.6c5.9,0.2,10.3-3.9,10.4-9.6
                    c0.5-5.5-3.6-10.4-9.1-10.8c-5.5-0.5-10.4,3.6-10.8,9.1c0,0.5,0,0.9,0,1.4c-0.2,5.3,4,9.8,9.3,10
                    C71.6,134.6,71.7,134.6,71.8,134.6z"/>
            </g>
        </svg>
        <table>
            <tr>
                <td style="text-align: left"><b>Python version:</b></td>
                <td style="text-align: left"><b>3.10.8</b></td>
            </tr>
            <tr>
                <td style="text-align: left"><b>Ray version:</b></td>
                <td style="text-align: left"><b> 2.2.0</b></td>
            </tr>

        </table>
    </div>
</div>




If we want to use Ray Tune we need to slightly change the definition of the train() function. Since Ray does not have any knowledge about the progress of the trial we need to report the training progress to Tune. For that purpose we can use the session.report() function. Furthermore the train function needs to accept a dictionary as input which contains the chosen hyperparameters. 


```python
def train_tune(config):
    epochs = config["epochs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(train_dataloader, model, loss_fn, optimizer, device)
        accuracy = test(val_dataloader, model, loss_fn, device)
        session.report(metrics={"mean_accuracy": accuracy, "epoch": t + 1})
```

Now that we have a compatible function for training We are almost ready to create a tuner. A tuner is used in Ray to construct and execute an HPO search. 

Before we define the resources which are vailable for each trial. In the standard configuration Ray assigns 1 cpu core to each trial. In our case we will assign 2 cpu cores and 1 gpu core for the trial if available. Otherwise only the cpu cores will be used. On my machine with 16 threads available this will lead to 8 concurrent trails. With tune.with_ressources() we take the train_tune() function and create a new function which respects the resource assignments.

Afterwards we need to define the search space of the parameters. We are going to do HPO for the parameters "learning rate" and "weight decay". The variable "epochs" is also defined in the config dictionary but it is a fixed value and will therefore not be altered by Ray. The other two are defined as distributions and therefore the algorithm will sample combinations from these distributions.

Now that we have defined the resources each trial can use and the search space (parameter space) which should be exploited we can define the tuner. The tuner takes the trainable function, the parameter space and a TuneConfig. TuneConfig defines how the HPO should be executed, e.g. which metric to optimize for (and whether it should be minimized or maximized) and the number of samples.

Now we can start the HPO. Ray will show in real time the state of the trials and usage of resources.


```python
# Define ressources for each trial
resources = {"cpu":2, "gpu":1} if torch.cuda.is_available() else {"cpu":2}
trainable = tune.with_resources(train_tune, resources=resources) 

# Define the properties of the search space
config = {
    "epochs": 10,
    "lr": tune.uniform(1e-1,1e-5),
    "weight_decay": tune.uniform(1e-2,1e-6)
}

# Create the HPO tuner
tuner = tune.Tuner(
    trainable,
    param_space = config,
    tune_config = tune.TuneConfig(
        metric = "mean_accuracy",
        mode = "max",
        num_samples = 8,
    )
)

# Start the tuner
results = tuner.fit()
```


<div class="tuneStatus">
  <div style="display: flex;flex-direction: row">
    <div style="display: flex;flex-direction: column;">
      <h3>Tune Status</h3>
      <table>
<tbody>
<tr><td>Current time:</td><td>2023-01-09 15:44:46</td></tr>
<tr><td>Running for: </td><td>00:04:01.50        </td></tr>
<tr><td>Memory:      </td><td>12.7/30.6 GiB      </td></tr>
</tbody>
</table>
    </div>
    <div class="vDivider"></div>
    <div class="systemInfo">
      <h3>System Info</h3>
      Using FIFO scheduling algorithm.<br>Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/10.29 GiB heap, 0.0/5.14 GiB objects
    </div>

  </div>
  <div class="hDivider"></div>
  <div class="trialStatus">
    <h3>Trial Status</h3>
    <table>
<thead>
<tr><th>Trial name            </th><th>status    </th><th>loc                  </th><th style="text-align: right;">        lr</th><th style="text-align: right;">  weight_decay</th><th style="text-align: right;">    acc</th><th style="text-align: right;">  iter</th><th style="text-align: right;">  total time (s)</th><th style="text-align: right;">  epoch</th></tr>
</thead>
<tbody>
<tr><td>train_tune_99951_00000</td><td>TERMINATED</td><td>192.168.188.20:849409</td><td style="text-align: right;">0.0536606 </td><td style="text-align: right;">   0.00716843 </td><td style="text-align: right;">77.3333</td><td style="text-align: right;">    10</td><td style="text-align: right;">         231.057</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_99951_00001</td><td>TERMINATED</td><td>192.168.188.20:849487</td><td style="text-align: right;">0.0145171 </td><td style="text-align: right;">   0.00995262 </td><td style="text-align: right;">83.0833</td><td style="text-align: right;">    10</td><td style="text-align: right;">         225.55 </td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_99951_00002</td><td>TERMINATED</td><td>192.168.188.20:849503</td><td style="text-align: right;">0.00306243</td><td style="text-align: right;">   0.00828528 </td><td style="text-align: right;">79.7   </td><td style="text-align: right;">    10</td><td style="text-align: right;">         223.122</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_99951_00003</td><td>TERMINATED</td><td>192.168.188.20:849556</td><td style="text-align: right;">0.023809  </td><td style="text-align: right;">   0.0079757  </td><td style="text-align: right;">84.6417</td><td style="text-align: right;">    10</td><td style="text-align: right;">         225.27 </td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_99951_00004</td><td>TERMINATED</td><td>192.168.188.20:849609</td><td style="text-align: right;">0.0509506 </td><td style="text-align: right;">   0.00874976 </td><td style="text-align: right;">84.5833</td><td style="text-align: right;">    10</td><td style="text-align: right;">         227.019</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_99951_00005</td><td>TERMINATED</td><td>192.168.188.20:849685</td><td style="text-align: right;">0.027261  </td><td style="text-align: right;">   0.0058419  </td><td style="text-align: right;">85.4333</td><td style="text-align: right;">    10</td><td style="text-align: right;">         223.744</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_99951_00006</td><td>TERMINATED</td><td>192.168.188.20:849758</td><td style="text-align: right;">0.0468367 </td><td style="text-align: right;">   0.000406038</td><td style="text-align: right;">87.775 </td><td style="text-align: right;">    10</td><td style="text-align: right;">         226.475</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_99951_00007</td><td>TERMINATED</td><td>192.168.188.20:849941</td><td style="text-align: right;">0.0621856 </td><td style="text-align: right;">   0.00992793 </td><td style="text-align: right;">84.4833</td><td style="text-align: right;">    10</td><td style="text-align: right;">         221.646</td><td style="text-align: right;">     10</td></tr>
</tbody>
</table>
  </div>
</div>
<style>
.tuneStatus {
  color: var(--jp-ui-font-color1);
}
.tuneStatus .systemInfo {
  display: flex;
  flex-direction: column;
}
.tuneStatus td {
  white-space: nowrap;
}
.tuneStatus .trialStatus {
  display: flex;
  flex-direction: column;
}
.tuneStatus h3 {
  font-weight: bold;
}
.tuneStatus .hDivider {
  border-bottom-width: var(--jp-border-width);
  border-bottom-color: var(--jp-border-color0);
  border-bottom-style: solid;
}
.tuneStatus .vDivider {
  border-left-width: var(--jp-border-width);
  border-left-color: var(--jp-border-color0);
  border-left-style: solid;
  margin: 0.5em 1em 0.5em 1em;
}
</style>



    2023-01-09 15:40:50,719	WARNING worker.py:1851 -- Warning: The actor ImplicitFunc is very large (45 MiB). Check that its definition is not implicitly capturing a large array or other object in scope. Tip: use ray.put() to put large objects in the Ray object store.
    2023-01-09 15:40:51,242	WARNING util.py:244 -- The `start_trial` operation took 1.045 s, which may be a performance bottleneck.
    2023-01-09 15:40:54,888	WARNING util.py:244 -- The `start_trial` operation took 0.513 s, which may be a performance bottleneck.
    2023-01-09 15:40:55,891	WARNING util.py:244 -- The `start_trial` operation took 0.536 s, which may be a performance bottleneck.
    2023-01-09 15:41:00,472	WARNING util.py:244 -- The `start_trial` operation took 0.604 s, which may be a performance bottleneck.



<div class="trialProgress">
  <h3>Trial Progress</h3>
  <table>
<thead>
<tr><th>Trial name            </th><th>date               </th><th>done  </th><th>episodes_total  </th><th style="text-align: right;">  epoch</th><th>experiment_id                   </th><th>experiment_tag                 </th><th>hostname  </th><th style="text-align: right;">  iterations_since_restore</th><th style="text-align: right;">  mean_accuracy</th><th>node_ip       </th><th style="text-align: right;">   pid</th><th style="text-align: right;">  time_since_restore</th><th style="text-align: right;">  time_this_iter_s</th><th style="text-align: right;">  time_total_s</th><th style="text-align: right;">  timestamp</th><th style="text-align: right;">  timesteps_since_restore</th><th>timesteps_total  </th><th style="text-align: right;">  training_iteration</th><th style="text-align: right;">   trial_id</th><th style="text-align: right;">  warmup_time</th></tr>
</thead>
<tbody>
<tr><td>train_tune_99951_00000</td><td>2023-01-09_15-44-44</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>47dd7dee9e554676880730e470d77d0a</td><td>0_lr=0.0537,weight_decay=0.0072</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        77.3333</td><td>192.168.188.20</td><td style="text-align: right;">849409</td><td style="text-align: right;">             231.057</td><td style="text-align: right;">           23.1407</td><td style="text-align: right;">       231.057</td><td style="text-align: right;"> 1673275484</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td style="text-align: right;">99951_00000</td><td style="text-align: right;">   0.00394893</td></tr>
<tr><td>train_tune_99951_00001</td><td>2023-01-09_15-44-43</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>7c009618c5224c01990321c801cb6549</td><td>1_lr=0.0145,weight_decay=0.0100</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        83.0833</td><td>192.168.188.20</td><td style="text-align: right;">849487</td><td style="text-align: right;">             225.55 </td><td style="text-align: right;">           23.6519</td><td style="text-align: right;">       225.55 </td><td style="text-align: right;"> 1673275483</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td style="text-align: right;">99951_00001</td><td style="text-align: right;">   0.00502801</td></tr>
<tr><td>train_tune_99951_00002</td><td>2023-01-09_15-44-41</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>dc218997946444e087406ef9f125b84e</td><td>2_lr=0.0031,weight_decay=0.0083</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        79.7   </td><td>192.168.188.20</td><td style="text-align: right;">849503</td><td style="text-align: right;">             223.122</td><td style="text-align: right;">           22.3263</td><td style="text-align: right;">       223.122</td><td style="text-align: right;"> 1673275481</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td style="text-align: right;">99951_00002</td><td style="text-align: right;">   0.00590062</td></tr>
<tr><td>train_tune_99951_00003</td><td>2023-01-09_15-44-44</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>f8d8c367498c4e6495b27606c7d5946b</td><td>3_lr=0.0238,weight_decay=0.0080</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        84.6417</td><td>192.168.188.20</td><td style="text-align: right;">849556</td><td style="text-align: right;">             225.27 </td><td style="text-align: right;">           23.7308</td><td style="text-align: right;">       225.27 </td><td style="text-align: right;"> 1673275484</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td style="text-align: right;">99951_00003</td><td style="text-align: right;">   0.00627685</td></tr>
<tr><td>train_tune_99951_00004</td><td>2023-01-09_15-44-46</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>3e560a2d8138468bb12d2e53617a1af7</td><td>4_lr=0.0510,weight_decay=0.0087</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        84.5833</td><td>192.168.188.20</td><td style="text-align: right;">849609</td><td style="text-align: right;">             227.019</td><td style="text-align: right;">           22.287 </td><td style="text-align: right;">       227.019</td><td style="text-align: right;"> 1673275486</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td style="text-align: right;">99951_00004</td><td style="text-align: right;">   0.00506282</td></tr>
<tr><td>train_tune_99951_00005</td><td>2023-01-09_15-44-43</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>25e6d4ff9ffc478887ced4eda58976d0</td><td>5_lr=0.0273,weight_decay=0.0058</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        85.4333</td><td>192.168.188.20</td><td style="text-align: right;">849685</td><td style="text-align: right;">             223.744</td><td style="text-align: right;">           22.5407</td><td style="text-align: right;">       223.744</td><td style="text-align: right;"> 1673275483</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td style="text-align: right;">99951_00005</td><td style="text-align: right;">   0.00806093</td></tr>
<tr><td>train_tune_99951_00006</td><td>2023-01-09_15-44-46</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>3c653f61616048e098c9ed98e7044f9b</td><td>6_lr=0.0468,weight_decay=0.0004</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        87.775 </td><td>192.168.188.20</td><td style="text-align: right;">849758</td><td style="text-align: right;">             226.475</td><td style="text-align: right;">           22.0158</td><td style="text-align: right;">       226.475</td><td style="text-align: right;"> 1673275486</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td style="text-align: right;">99951_00006</td><td style="text-align: right;">   0.00750184</td></tr>
<tr><td>train_tune_99951_00007</td><td>2023-01-09_15-44-46</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>0e8053ed08ed4a5aa8542b794335a239</td><td>7_lr=0.0622,weight_decay=0.0099</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        84.4833</td><td>192.168.188.20</td><td style="text-align: right;">849941</td><td style="text-align: right;">             221.646</td><td style="text-align: right;">           22.2945</td><td style="text-align: right;">       221.646</td><td style="text-align: right;"> 1673275486</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td style="text-align: right;">99951_00007</td><td style="text-align: right;">   0.00741696</td></tr>
</tbody>
</table>
</div>
<style>
.trialProgress {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
}
.trialProgress h3 {
  font-weight: bold;
}
.trialProgress td {
  white-space: nowrap;
}
</style>



    2023-01-09 15:44:46,919	INFO tune.py:762 -- Total run time: 242.24 seconds (241.49 seconds for the tuning loop).



```python
print(results.get_best_result())

ax = None
dfs = {result.log_dir: result.metrics_dataframe for result in results}
for d in dfs.values():
    ax = d.plot(ax=ax, y="mean_accuracy", x="epoch", legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean accuracy")
```

    Result(metrics={'mean_accuracy': 87.775, 'epoch': 10, 'done': True, 'trial_id': '99951_00006', 'experiment_tag': '6_lr=0.0468,weight_decay=0.0004'}, error=None, log_dir=PosixPath('/home/raffi/ray_results/train_tune_2023-01-09_15-40-44/train_tune_99951_00006_6_lr=0.0468,weight_decay=0.0004_2023-01-09_15-40-55'))





    Text(0, 0.5, 'Mean accuracy')




    
![png](Ray_Tune_FashionMNIST_files/Ray_Tune_FashionMNIST_29_2.png)
    


## Hyperparameter optimization (ASAH Scheduler)

Now we are going to make the example slightly harder by integrating the ASAH scheduler. This scheduler will try to stop non-promising trials in order to save ressources. This allows us to choose a higher number of samples without increasing the training time. 

We can reuse the train_tune() function and the config from before. Afterwards we need to define the scheduler with the correct arguments. 

Now we only need to add the scheduler to the TuneConfig and afterwards we can start the HPO.

**HINT**: The ASAH scheduler can also take "metric" and "mode" as input. If you already defined this in the TuneConfig do **NOT** redefine it.


```python
scheduler = ASHAScheduler(
    max_t=config["epochs"], # Max time per trial
    time_attr="training_iteration", # Which metric is used as measurement for "time"
    grace_period=2
    )

tuner = tune.Tuner(
    trainable,
    param_space = config,
    tune_config = tune.TuneConfig(
        metric = "mean_accuracy",
        mode = "max",
        num_samples = 16,
        scheduler=scheduler,
    )
)

results_asah = tuner.fit()
```


<div class="tuneStatus">
  <div style="display: flex;flex-direction: row">
    <div style="display: flex;flex-direction: column;">
      <h3>Tune Status</h3>
      <table>
<tbody>
<tr><td>Current time:</td><td>2023-01-09 15:49:14</td></tr>
<tr><td>Running for: </td><td>00:04:26.21        </td></tr>
<tr><td>Memory:      </td><td>13.9/30.6 GiB      </td></tr>
</tbody>
</table>
    </div>
    <div class="vDivider"></div>
    <div class="systemInfo">
      <h3>System Info</h3>
      Using AsyncHyperBand: num_stopped=16<br>Bracket: Iter 8.000: 86.49166666666666 | Iter 2.000: 82.80833333333334<br>Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/10.29 GiB heap, 0.0/5.14 GiB objects
    </div>

  </div>
  <div class="hDivider"></div>
  <div class="trialStatus">
    <h3>Trial Status</h3>
    <table>
<thead>
<tr><th>Trial name            </th><th>status    </th><th>loc                  </th><th style="text-align: right;">        lr</th><th style="text-align: right;">  weight_decay</th><th style="text-align: right;">    acc</th><th style="text-align: right;">  iter</th><th style="text-align: right;">  total time (s)</th><th style="text-align: right;">  epoch</th></tr>
</thead>
<tbody>
<tr><td>train_tune_2a47e_00000</td><td>TERMINATED</td><td>192.168.188.20:850436</td><td style="text-align: right;">0.0803609 </td><td style="text-align: right;">   0.00447683 </td><td style="text-align: right;">85.8333</td><td style="text-align: right;">    10</td><td style="text-align: right;">        195.027 </td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_2a47e_00001</td><td>TERMINATED</td><td>192.168.188.20:850512</td><td style="text-align: right;">0.0442203 </td><td style="text-align: right;">   0.00558784 </td><td style="text-align: right;">82.625 </td><td style="text-align: right;">     2</td><td style="text-align: right;">         45.065 </td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00002</td><td>TERMINATED</td><td>192.168.188.20:850539</td><td style="text-align: right;">0.0412758 </td><td style="text-align: right;">   0.00899764 </td><td style="text-align: right;">79.025 </td><td style="text-align: right;">     2</td><td style="text-align: right;">         45.0777</td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00003</td><td>TERMINATED</td><td>192.168.188.20:850571</td><td style="text-align: right;">0.0373015 </td><td style="text-align: right;">   0.00565958 </td><td style="text-align: right;">81.15  </td><td style="text-align: right;">     2</td><td style="text-align: right;">         45.488 </td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00004</td><td>TERMINATED</td><td>192.168.188.20:850645</td><td style="text-align: right;">0.013251  </td><td style="text-align: right;">   0.00842493 </td><td style="text-align: right;">78.25  </td><td style="text-align: right;">     2</td><td style="text-align: right;">         46.0169</td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00005</td><td>TERMINATED</td><td>192.168.188.20:850719</td><td style="text-align: right;">0.0132757 </td><td style="text-align: right;">   0.00130614 </td><td style="text-align: right;">79.0583</td><td style="text-align: right;">     2</td><td style="text-align: right;">         45.2109</td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00006</td><td>TERMINATED</td><td>192.168.188.20:850900</td><td style="text-align: right;">0.0956718 </td><td style="text-align: right;">   0.00624477 </td><td style="text-align: right;">85.4833</td><td style="text-align: right;">     8</td><td style="text-align: right;">        160.435 </td><td style="text-align: right;">      8</td></tr>
<tr><td>train_tune_2a47e_00007</td><td>TERMINATED</td><td>192.168.188.20:850916</td><td style="text-align: right;">0.0840716 </td><td style="text-align: right;">   0.00475829 </td><td style="text-align: right;">82.75  </td><td style="text-align: right;">     2</td><td style="text-align: right;">         46.7948</td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00008</td><td>TERMINATED</td><td>192.168.188.20:850512</td><td style="text-align: right;">0.0886306 </td><td style="text-align: right;">   0.00589293 </td><td style="text-align: right;">81.2083</td><td style="text-align: right;">     2</td><td style="text-align: right;">         46.5557</td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00009</td><td>TERMINATED</td><td>192.168.188.20:850539</td><td style="text-align: right;">0.0789423 </td><td style="text-align: right;">   0.00280516 </td><td style="text-align: right;">81.95  </td><td style="text-align: right;">     2</td><td style="text-align: right;">         47.061 </td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00010</td><td>TERMINATED</td><td>192.168.188.20:850571</td><td style="text-align: right;">0.015722  </td><td style="text-align: right;">   0.00112272 </td><td style="text-align: right;">79.725 </td><td style="text-align: right;">     2</td><td style="text-align: right;">         46.3901</td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00011</td><td>TERMINATED</td><td>192.168.188.20:850719</td><td style="text-align: right;">0.0101423 </td><td style="text-align: right;">   0.000511431</td><td style="text-align: right;">76.7   </td><td style="text-align: right;">     2</td><td style="text-align: right;">         46.8588</td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00012</td><td>TERMINATED</td><td>192.168.188.20:850645</td><td style="text-align: right;">0.00622107</td><td style="text-align: right;">   0.00844181 </td><td style="text-align: right;">69.35  </td><td style="text-align: right;">     2</td><td style="text-align: right;">         47.2438</td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00013</td><td>TERMINATED</td><td>192.168.188.20:850916</td><td style="text-align: right;">0.0210617 </td><td style="text-align: right;">   0.00870492 </td><td style="text-align: right;">80.2917</td><td style="text-align: right;">     2</td><td style="text-align: right;">         44.8909</td><td style="text-align: right;">      2</td></tr>
<tr><td>train_tune_2a47e_00014</td><td>TERMINATED</td><td>192.168.188.20:850512</td><td style="text-align: right;">0.0567571 </td><td style="text-align: right;">   0.0032487  </td><td style="text-align: right;">87.0167</td><td style="text-align: right;">    10</td><td style="text-align: right;">        157.554 </td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_2a47e_00015</td><td>TERMINATED</td><td>192.168.188.20:850539</td><td style="text-align: right;">0.0942973 </td><td style="text-align: right;">   0.00194781 </td><td style="text-align: right;">85.2917</td><td style="text-align: right;">    10</td><td style="text-align: right;">        156.683 </td><td style="text-align: right;">     10</td></tr>
</tbody>
</table>
  </div>
</div>
<style>
.tuneStatus {
  color: var(--jp-ui-font-color1);
}
.tuneStatus .systemInfo {
  display: flex;
  flex-direction: column;
}
.tuneStatus td {
  white-space: nowrap;
}
.tuneStatus .trialStatus {
  display: flex;
  flex-direction: column;
}
.tuneStatus h3 {
  font-weight: bold;
}
.tuneStatus .hDivider {
  border-bottom-width: var(--jp-border-width);
  border-bottom-color: var(--jp-border-color0);
  border-bottom-style: solid;
}
.tuneStatus .vDivider {
  border-left-width: var(--jp-border-width);
  border-left-color: var(--jp-border-color0);
  border-left-style: solid;
  margin: 0.5em 1em 0.5em 1em;
}
</style>



    2023-01-09 15:44:57,968	WARNING worker.py:1851 -- Warning: The actor ImplicitFunc is very large (45 MiB). Check that its definition is not implicitly capturing a large array or other object in scope. Tip: use ray.put() to put large objects in the Ray object store.
    2023-01-09 15:44:58,134	WARNING util.py:244 -- The `start_trial` operation took 1.016 s, which may be a performance bottleneck.
    2023-01-09 15:45:02,636	WARNING util.py:244 -- The `start_trial` operation took 0.548 s, which may be a performance bottleneck.
    2023-01-09 15:45:03,165	WARNING util.py:244 -- The `start_trial` operation took 0.528 s, which may be a performance bottleneck.
    2023-01-09 15:45:07,245	WARNING util.py:244 -- The `start_trial` operation took 0.519 s, which may be a performance bottleneck.
    2023-01-09 15:45:07,932	WARNING util.py:244 -- The `start_trial` operation took 0.685 s, which may be a performance bottleneck.



<div class="trialProgress">
  <h3>Trial Progress</h3>
  <table>
<thead>
<tr><th>Trial name            </th><th>date               </th><th>done  </th><th>episodes_total  </th><th style="text-align: right;">  epoch</th><th>experiment_id                   </th><th>hostname  </th><th style="text-align: right;">  iterations_since_restore</th><th style="text-align: right;">  mean_accuracy</th><th>node_ip       </th><th style="text-align: right;">   pid</th><th style="text-align: right;">  time_since_restore</th><th style="text-align: right;">  time_this_iter_s</th><th style="text-align: right;">  time_total_s</th><th style="text-align: right;">  timestamp</th><th style="text-align: right;">  timesteps_since_restore</th><th>timesteps_total  </th><th style="text-align: right;">  training_iteration</th><th>trial_id   </th><th style="text-align: right;">  warmup_time</th></tr>
</thead>
<tbody>
<tr><td>train_tune_2a47e_00000</td><td>2023-01-09_15-48-15</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>d2f52c8000014e1c8cec47180a5f1ed9</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        85.8333</td><td>192.168.188.20</td><td style="text-align: right;">850436</td><td style="text-align: right;">            195.027 </td><td style="text-align: right;">           15.5158</td><td style="text-align: right;">      195.027 </td><td style="text-align: right;"> 1673275695</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>2a47e_00000</td><td style="text-align: right;">   0.00615978</td></tr>
<tr><td>train_tune_2a47e_00001</td><td>2023-01-09_15-45-50</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>4568aae5fd2e42c5ad9bb1b48a7a522e</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        82.625 </td><td>192.168.188.20</td><td style="text-align: right;">850512</td><td style="text-align: right;">             45.065 </td><td style="text-align: right;">           23.1897</td><td style="text-align: right;">       45.065 </td><td style="text-align: right;"> 1673275550</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00001</td><td style="text-align: right;">   0.00562286</td></tr>
<tr><td>train_tune_2a47e_00002</td><td>2023-01-09_15-45-50</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>aa21989de996413c87c4d933ef8f9ea5</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        79.025 </td><td>192.168.188.20</td><td style="text-align: right;">850539</td><td style="text-align: right;">             45.0777</td><td style="text-align: right;">           22.7481</td><td style="text-align: right;">       45.0777</td><td style="text-align: right;"> 1673275550</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00002</td><td style="text-align: right;">   0.00646687</td></tr>
<tr><td>train_tune_2a47e_00003</td><td>2023-01-09_15-45-51</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>30c33931842e4ae5a2cf204b6a7ffe12</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        81.15  </td><td>192.168.188.20</td><td style="text-align: right;">850571</td><td style="text-align: right;">             45.488 </td><td style="text-align: right;">           23.0182</td><td style="text-align: right;">       45.488 </td><td style="text-align: right;"> 1673275551</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00003</td><td style="text-align: right;">   0.00507331</td></tr>
<tr><td>train_tune_2a47e_00004</td><td>2023-01-09_15-45-52</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>9e0dbcdb388b408f9c5d0dec90f8dc6f</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        78.25  </td><td>192.168.188.20</td><td style="text-align: right;">850645</td><td style="text-align: right;">             46.0169</td><td style="text-align: right;">           22.7018</td><td style="text-align: right;">       46.0169</td><td style="text-align: right;"> 1673275552</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00004</td><td style="text-align: right;">   0.00560999</td></tr>
<tr><td>train_tune_2a47e_00005</td><td>2023-01-09_15-45-51</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>086a7aca978647f0a6003379ddc26448</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        79.0583</td><td>192.168.188.20</td><td style="text-align: right;">850719</td><td style="text-align: right;">             45.2109</td><td style="text-align: right;">           22.788 </td><td style="text-align: right;">       45.2109</td><td style="text-align: right;"> 1673275551</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00005</td><td style="text-align: right;">   0.00536489</td></tr>
<tr><td>train_tune_2a47e_00006</td><td>2023-01-09_15-47-53</td><td>True  </td><td>                </td><td style="text-align: right;">      8</td><td>593cdc2998944ee89b335daba7aada45</td><td>fedora    </td><td style="text-align: right;">                         8</td><td style="text-align: right;">        85.4833</td><td>192.168.188.20</td><td style="text-align: right;">850900</td><td style="text-align: right;">            160.435 </td><td style="text-align: right;">           16.8421</td><td style="text-align: right;">      160.435 </td><td style="text-align: right;"> 1673275673</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   8</td><td>2a47e_00006</td><td style="text-align: right;">   0.006392  </td></tr>
<tr><td>train_tune_2a47e_00007</td><td>2023-01-09_15-45-59</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>5e4a930d8bc14d8da2de2ffa8ddac9c6</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        82.75  </td><td>192.168.188.20</td><td style="text-align: right;">850916</td><td style="text-align: right;">             46.7948</td><td style="text-align: right;">           22.9463</td><td style="text-align: right;">       46.7948</td><td style="text-align: right;"> 1673275559</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00007</td><td style="text-align: right;">   0.00485849</td></tr>
<tr><td>train_tune_2a47e_00008</td><td>2023-01-09_15-46-36</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>4568aae5fd2e42c5ad9bb1b48a7a522e</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        81.2083</td><td>192.168.188.20</td><td style="text-align: right;">850512</td><td style="text-align: right;">             46.5557</td><td style="text-align: right;">           23.0082</td><td style="text-align: right;">       46.5557</td><td style="text-align: right;"> 1673275596</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00008</td><td style="text-align: right;">   0.00562286</td></tr>
<tr><td>train_tune_2a47e_00009</td><td>2023-01-09_15-46-37</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>aa21989de996413c87c4d933ef8f9ea5</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        81.95  </td><td>192.168.188.20</td><td style="text-align: right;">850539</td><td style="text-align: right;">             47.061 </td><td style="text-align: right;">           23.9621</td><td style="text-align: right;">       47.061 </td><td style="text-align: right;"> 1673275597</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00009</td><td style="text-align: right;">   0.00646687</td></tr>
<tr><td>train_tune_2a47e_00010</td><td>2023-01-09_15-46-37</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>30c33931842e4ae5a2cf204b6a7ffe12</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        79.725 </td><td>192.168.188.20</td><td style="text-align: right;">850571</td><td style="text-align: right;">             46.3901</td><td style="text-align: right;">           23.5714</td><td style="text-align: right;">       46.3901</td><td style="text-align: right;"> 1673275597</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00010</td><td style="text-align: right;">   0.00507331</td></tr>
<tr><td>train_tune_2a47e_00011</td><td>2023-01-09_15-46-38</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>086a7aca978647f0a6003379ddc26448</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        76.7   </td><td>192.168.188.20</td><td style="text-align: right;">850719</td><td style="text-align: right;">             46.8588</td><td style="text-align: right;">           23.7951</td><td style="text-align: right;">       46.8588</td><td style="text-align: right;"> 1673275598</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00011</td><td style="text-align: right;">   0.00536489</td></tr>
<tr><td>train_tune_2a47e_00012</td><td>2023-01-09_15-46-39</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>9e0dbcdb388b408f9c5d0dec90f8dc6f</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        69.35  </td><td>192.168.188.20</td><td style="text-align: right;">850645</td><td style="text-align: right;">             47.2438</td><td style="text-align: right;">           23.4869</td><td style="text-align: right;">       47.2438</td><td style="text-align: right;"> 1673275599</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00012</td><td style="text-align: right;">   0.00560999</td></tr>
<tr><td>train_tune_2a47e_00013</td><td>2023-01-09_15-46-44</td><td>True  </td><td>                </td><td style="text-align: right;">      2</td><td>5e4a930d8bc14d8da2de2ffa8ddac9c6</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        80.2917</td><td>192.168.188.20</td><td style="text-align: right;">850916</td><td style="text-align: right;">             44.8909</td><td style="text-align: right;">           21.8882</td><td style="text-align: right;">       44.8909</td><td style="text-align: right;"> 1673275604</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   2</td><td>2a47e_00013</td><td style="text-align: right;">   0.00485849</td></tr>
<tr><td>train_tune_2a47e_00014</td><td>2023-01-09_15-49-14</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>4568aae5fd2e42c5ad9bb1b48a7a522e</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        87.0167</td><td>192.168.188.20</td><td style="text-align: right;">850512</td><td style="text-align: right;">            157.554 </td><td style="text-align: right;">           13.7587</td><td style="text-align: right;">      157.554 </td><td style="text-align: right;"> 1673275754</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>2a47e_00014</td><td style="text-align: right;">   0.00562286</td></tr>
<tr><td>train_tune_2a47e_00015</td><td>2023-01-09_15-49-14</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>aa21989de996413c87c4d933ef8f9ea5</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        85.2917</td><td>192.168.188.20</td><td style="text-align: right;">850539</td><td style="text-align: right;">            156.683 </td><td style="text-align: right;">           13.6614</td><td style="text-align: right;">      156.683 </td><td style="text-align: right;"> 1673275754</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>2a47e_00015</td><td style="text-align: right;">   0.00646687</td></tr>
</tbody>
</table>
</div>
<style>
.trialProgress {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
}
.trialProgress h3 {
  font-weight: bold;
}
.trialProgress td {
  white-space: nowrap;
}
</style>



    2023-01-09 15:49:14,517	INFO tune.py:762 -- Total run time: 267.09 seconds (266.20 seconds for the tuning loop).



```python
print(results_asah.get_best_result())

ax = None
dfs = {result.log_dir: result.metrics_dataframe for result in results_asah}
for d in dfs.values():
    ax = d.plot(ax=ax, y="mean_accuracy", x="epoch", legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean accuracy")
```

    Result(metrics={'mean_accuracy': 87.01666666666667, 'epoch': 10, 'done': True, 'trial_id': '2a47e_00014', 'experiment_tag': '14_lr=0.0568,weight_decay=0.0032'}, error=None, log_dir=PosixPath('/home/raffi/ray_results/train_tune_2023-01-09_15-44-47/train_tune_2a47e_00014_14_lr=0.0568,weight_decay=0.0032_2023-01-09_15-46-36'))





    Text(0, 0.5, 'Mean accuracy')




    
![png](Ray_Tune_FashionMNIST_files/Ray_Tune_FashionMNIST_33_2.png)
    


In the graph we can see that indeed trials were stopped by ASAH.

## Hyperparameter optimization (PBT)

<img src="./images/tune_pbt.png" 
     align="center" 
     width="600" />

*Ray Framework Overview (https://docs.ray.io/en/latest/index.html)*

One of the most interesting HPO algorithms in Ray Tune is the Population Based Training (PBT). In PBT we do not see each trial as independent but we rather try to increase the performance of the whole population of trials. After some epochs PBT will replace bad performing trials with good performing ones and perturb their parameters. Similar to the ASAH scheduler this will ensure that non-promising trials are stopped early on and that the search focuses on the promising parts in the search space. 

Ray Tune contains a distributed implementation of this algorithm. Since PBT will replace the bad with the good trials we need to somehow store the state of the good trials. This can be achieved by using checkpoints in Ray. The code below shows the modified train_tune() function.

Let's first look at the added lines in the training loop. We first create a folder for the checkpoint with os.makedirs(). Afterwards we use the torch.save() function to store the models parameters and the current epoch. We can now create a checkpoint by using the Checkpoint.from_directory() function. This checkpoint can be added to the session.report() function which we already used before. Now a checkpoint containing the model's state will be created in each iteration. For large epoch number the checkpoint frequency should be lower to avoid unnecessary overhead.

If a bad trial is stopped Tune will create a new trial and run the training function with the perturbed hyperparameters. It will furthermore copy the checkpoint of the better performing trial. Therefore we need to check whether a checkpoint is available at the beginning of the training. If there is no checkpoint available we now that this trial is new and was not replace. If there is a checkpoint available then the trial was replaced and we need to load the state of the model. Furthermore we need to set the current epoch to the correct value.


```python
def train_tune_pbt(config):
    step = 0
    epochs = config["epochs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if session.get_checkpoint():
        loaded_checkpoint = session.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            path = os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["model_state"])
            step = checkpoint["epoch"] + 1

    for t in range(step, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(train_dataloader, model, loss_fn, optimizer, device)
        accuracy = test(val_dataloader, model, loss_fn, device)

        os.makedirs("model", exist_ok=True)
        torch.save(
            {
                "epoch": t,
                "model_state": model.state_dict(),
            },
            "model/checkpoint.pt"
        )
        checkpoint = Checkpoint.from_directory("model")
        session.report(metrics={"mean_accuracy": accuracy, "epoch": t + 1}, checkpoint=checkpoint)
```

Now the process is similar to the examples before. We create a trainable function and assign the needed ressources. Instead of defining the ASAH scheduler as before we define the PBT scheduler and the tuner is created and started as done before.

**HINT**: The PBT scheduler can also take "metric" and "mode" as input. If you already defined this in the TuneConfig do **NOT** redefine it.


```python
# Define resources for each trial
resources = {"cpu":2, "gpu":1} if torch.cuda.is_available() else {"cpu":2}
trainable = tune.with_resources(train_tune_pbt, resources=resources) 

# Define the PBT algorithm
scheduler = PopulationBasedTraining(
    time_attr = "training_iteration",
    perturbation_interval = 2,
    hyperparam_mutations = {
        "lr": tune.uniform(1e-1,1e-5),
        "weight_decay": tune.uniform(1e-2,1e-6)
    },
)

# Create the HPO tuner
tuner = tune.Tuner(
    trainable,
    param_space = config,
    tune_config = tune.TuneConfig(
        metric = "mean_accuracy",
        mode = "max",
        num_samples = 8,
        scheduler=scheduler,
    )
)

# Run the tuner
results_pbt = tuner.fit()
```


<div class="tuneStatus">
  <div style="display: flex;flex-direction: row">
    <div style="display: flex;flex-direction: column;">
      <h3>Tune Status</h3>
      <table>
<tbody>
<tr><td>Current time:</td><td>2023-01-09 15:55:33</td></tr>
<tr><td>Running for: </td><td>00:06:17.83        </td></tr>
<tr><td>Memory:      </td><td>13.1/30.6 GiB      </td></tr>
</tbody>
</table>
    </div>
    <div class="vDivider"></div>
    <div class="systemInfo">
      <h3>System Info</h3>
      PopulationBasedTraining: 16 checkpoints, 3 perturbs<br>Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/10.29 GiB heap, 0.0/5.14 GiB objects
    </div>

  </div>
  <div class="hDivider"></div>
  <div class="trialStatus">
    <h3>Trial Status</h3>
    <table>
<thead>
<tr><th>Trial name                </th><th>status    </th><th>loc                  </th><th style="text-align: right;">       lr</th><th style="text-align: right;">  weight_decay</th><th style="text-align: right;">    acc</th><th style="text-align: right;">  iter</th><th style="text-align: right;">  total time (s)</th><th style="text-align: right;">  epoch</th></tr>
</thead>
<tbody>
<tr><td>train_tune_pbt_c9d37_00000</td><td>TERMINATED</td><td>192.168.188.20:852954</td><td style="text-align: right;">0.0386163</td><td style="text-align: right;">   0.00219223 </td><td style="text-align: right;">86.75  </td><td style="text-align: right;">    10</td><td style="text-align: right;">         287.351</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_pbt_c9d37_00001</td><td>TERMINATED</td><td>192.168.188.20:852768</td><td style="text-align: right;">0.0321803</td><td style="text-align: right;">   0.00460694 </td><td style="text-align: right;">86.2917</td><td style="text-align: right;">    10</td><td style="text-align: right;">         328.034</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_pbt_c9d37_00002</td><td>TERMINATED</td><td>192.168.188.20:851570</td><td style="text-align: right;">0.0255263</td><td style="text-align: right;">   0.00101877 </td><td style="text-align: right;">86.6167</td><td style="text-align: right;">    10</td><td style="text-align: right;">         332.429</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_pbt_c9d37_00003</td><td>TERMINATED</td><td>192.168.188.20:851643</td><td style="text-align: right;">0.016054 </td><td style="text-align: right;">   0.000548364</td><td style="text-align: right;">85.6833</td><td style="text-align: right;">    10</td><td style="text-align: right;">         335.07 </td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_pbt_c9d37_00004</td><td>TERMINATED</td><td>192.168.188.20:852605</td><td style="text-align: right;">0.0321803</td><td style="text-align: right;">   0.00274029 </td><td style="text-align: right;">82.35  </td><td style="text-align: right;">    10</td><td style="text-align: right;">         323.487</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_pbt_c9d37_00005</td><td>TERMINATED</td><td>192.168.188.20:851751</td><td style="text-align: right;">0.0243282</td><td style="text-align: right;">   0.00135175 </td><td style="text-align: right;">86.2833</td><td style="text-align: right;">    10</td><td style="text-align: right;">         330.257</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_pbt_c9d37_00006</td><td>TERMINATED</td><td>192.168.188.20:851825</td><td style="text-align: right;">0.0327867</td><td style="text-align: right;">   0.000448622</td><td style="text-align: right;">86.25  </td><td style="text-align: right;">    10</td><td style="text-align: right;">         328.115</td><td style="text-align: right;">     10</td></tr>
<tr><td>train_tune_pbt_c9d37_00007</td><td>TERMINATED</td><td>192.168.188.20:852005</td><td style="text-align: right;">0.0279731</td><td style="text-align: right;">   0.00356693 </td><td style="text-align: right;">85.7583</td><td style="text-align: right;">    10</td><td style="text-align: right;">         336.267</td><td style="text-align: right;">     10</td></tr>
</tbody>
</table>
  </div>
</div>
<style>
.tuneStatus {
  color: var(--jp-ui-font-color1);
}
.tuneStatus .systemInfo {
  display: flex;
  flex-direction: column;
}
.tuneStatus td {
  white-space: nowrap;
}
.tuneStatus .trialStatus {
  display: flex;
  flex-direction: column;
}
.tuneStatus h3 {
  font-weight: bold;
}
.tuneStatus .hDivider {
  border-bottom-width: var(--jp-border-width);
  border-bottom-color: var(--jp-border-color0);
  border-bottom-style: solid;
}
.tuneStatus .vDivider {
  border-left-width: var(--jp-border-width);
  border-left-color: var(--jp-border-color0);
  border-left-style: solid;
  margin: 0.5em 1em 0.5em 1em;
}
</style>



    2023-01-09 15:49:21,344	WARNING worker.py:1851 -- Warning: The actor ImplicitFunc is very large (45 MiB). Check that its definition is not implicitly capturing a large array or other object in scope. Tip: use ray.put() to put large objects in the Ray object store.
    2023-01-09 15:49:21,524	WARNING util.py:244 -- The `start_trial` operation took 1.051 s, which may be a performance bottleneck.
    2023-01-09 15:49:25,252	WARNING util.py:244 -- The `start_trial` operation took 0.519 s, which may be a performance bottleneck.
    2023-01-09 15:49:26,256	WARNING util.py:244 -- The `start_trial` operation took 0.520 s, which may be a performance bottleneck.
    2023-01-09 15:49:26,758	WARNING util.py:244 -- The `start_trial` operation took 0.501 s, which may be a performance bottleneck.
    2023-01-09 15:49:31,001	WARNING util.py:244 -- The `start_trial` operation took 0.612 s, which may be a performance bottleneck.



<div class="trialProgress">
  <h3>Trial Progress</h3>
  <table>
<thead>
<tr><th>Trial name                </th><th>date               </th><th>done  </th><th>episodes_total  </th><th style="text-align: right;">  epoch</th><th>experiment_id                   </th><th>experiment_tag                                                          </th><th>hostname  </th><th style="text-align: right;">  iterations_since_restore</th><th style="text-align: right;">  mean_accuracy</th><th>node_ip       </th><th style="text-align: right;">   pid</th><th>should_checkpoint  </th><th style="text-align: right;">  time_since_restore</th><th style="text-align: right;">  time_this_iter_s</th><th style="text-align: right;">  time_total_s</th><th style="text-align: right;">  timestamp</th><th style="text-align: right;">  timesteps_since_restore</th><th>timesteps_total  </th><th style="text-align: right;">  training_iteration</th><th>trial_id   </th><th style="text-align: right;">  warmup_time</th></tr>
</thead>
<tbody>
<tr><td>train_tune_pbt_c9d37_00000</td><td>2023-01-09_15-55-33</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>2a50c45ab9e54e938b91149a8870a41f</td><td>0_lr=0.0402,weight_decay=0.0038@perturbed[lr=0.0386,weight_decay=0.0022]</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        86.75  </td><td>192.168.188.20</td><td style="text-align: right;">852954</td><td>True               </td><td style="text-align: right;">             33.4527</td><td style="text-align: right;">           14.6259</td><td style="text-align: right;">       287.351</td><td style="text-align: right;"> 1673276133</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>c9d37_00000</td><td style="text-align: right;">   0.0266705 </td></tr>
<tr><td>train_tune_pbt_c9d37_00001</td><td>2023-01-09_15-55-05</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>2a50c45ab9e54e938b91149a8870a41f</td><td>1_lr=0.0288,weight_decay=0.0082@perturbed[lr=0.0322,weight_decay=0.0046]</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        86.2917</td><td>192.168.188.20</td><td style="text-align: right;">852768</td><td>True               </td><td style="text-align: right;">             72.4481</td><td style="text-align: right;">           35.5531</td><td style="text-align: right;">       328.034</td><td style="text-align: right;"> 1673276105</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>c9d37_00001</td><td style="text-align: right;">   0.0290251 </td></tr>
<tr><td>train_tune_pbt_c9d37_00002</td><td>2023-01-09_15-55-01</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>d0eb4e3030ba4625ab4af3b8255022a0</td><td>2_lr=0.0255,weight_decay=0.0010                                         </td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        86.6167</td><td>192.168.188.20</td><td style="text-align: right;">851570</td><td>True               </td><td style="text-align: right;">            332.429 </td><td style="text-align: right;">           38.0322</td><td style="text-align: right;">       332.429</td><td style="text-align: right;"> 1673276101</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>c9d37_00002</td><td style="text-align: right;">   0.00557184</td></tr>
<tr><td>train_tune_pbt_c9d37_00003</td><td>2023-01-09_15-55-04</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>3bb112fc72684f539140bacc2090dd11</td><td>3_lr=0.0161,weight_decay=0.0005                                         </td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        85.6833</td><td>192.168.188.20</td><td style="text-align: right;">851643</td><td>True               </td><td style="text-align: right;">            335.07  </td><td style="text-align: right;">           35.8159</td><td style="text-align: right;">       335.07 </td><td style="text-align: right;"> 1673276104</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>c9d37_00003</td><td style="text-align: right;">   0.00639296</td></tr>
<tr><td>train_tune_pbt_c9d37_00004</td><td>2023-01-09_15-55-09</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>2a50c45ab9e54e938b91149a8870a41f</td><td>4_lr=0.0406,weight_decay=0.0095@perturbed[lr=0.0322,weight_decay=0.0027]</td><td>fedora    </td><td style="text-align: right;">                         4</td><td style="text-align: right;">        82.35  </td><td>192.168.188.20</td><td style="text-align: right;">852605</td><td>True               </td><td style="text-align: right;">            142.695 </td><td style="text-align: right;">           30.9236</td><td style="text-align: right;">       323.487</td><td style="text-align: right;"> 1673276109</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>c9d37_00004</td><td style="text-align: right;">   0.0456536 </td></tr>
<tr><td>train_tune_pbt_c9d37_00005</td><td>2023-01-09_15-55-00</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>a55c25ac1e11433db72117ffc8dba908</td><td>5_lr=0.0243,weight_decay=0.0014                                         </td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        86.2833</td><td>192.168.188.20</td><td style="text-align: right;">851751</td><td>True               </td><td style="text-align: right;">            330.257 </td><td style="text-align: right;">           37.3682</td><td style="text-align: right;">       330.257</td><td style="text-align: right;"> 1673276100</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>c9d37_00005</td><td style="text-align: right;">   0.00917864</td></tr>
<tr><td>train_tune_pbt_c9d37_00006</td><td>2023-01-09_15-54-58</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>80d84d1305a94ca681673c2aabea0ca0</td><td>6_lr=0.0328,weight_decay=0.0004                                         </td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        86.25  </td><td>192.168.188.20</td><td style="text-align: right;">851825</td><td>True               </td><td style="text-align: right;">            328.115 </td><td style="text-align: right;">           36.4608</td><td style="text-align: right;">       328.115</td><td style="text-align: right;"> 1673276098</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>c9d37_00006</td><td style="text-align: right;">   0.00466776</td></tr>
<tr><td>train_tune_pbt_c9d37_00007</td><td>2023-01-09_15-55-11</td><td>True  </td><td>                </td><td style="text-align: right;">     10</td><td>64e2e219523344c69ba40ccbfb6293fb</td><td>7_lr=0.0280,weight_decay=0.0036                                         </td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        85.7583</td><td>192.168.188.20</td><td style="text-align: right;">852005</td><td>True               </td><td style="text-align: right;">            336.267 </td><td style="text-align: right;">           29.3908</td><td style="text-align: right;">       336.267</td><td style="text-align: right;"> 1673276111</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>c9d37_00007</td><td style="text-align: right;">   0.00582099</td></tr>
</tbody>
</table>
</div>
<style>
.trialProgress {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
}
.trialProgress h3 {
  font-weight: bold;
}
.trialProgress td {
  white-space: nowrap;
}
</style>



    2023-01-09 15:50:13,684	INFO pbt.py:646 -- [pbt]: no checkpoint for trial. Skip exploit for Trial train_tune_pbt_c9d37_00001
    2023-01-09 15:50:15,557	INFO pbt.py:646 -- [pbt]: no checkpoint for trial. Skip exploit for Trial train_tune_pbt_c9d37_00003
    2023-01-09 15:50:16,495	INFO pbt.py:646 -- [pbt]: no checkpoint for trial. Skip exploit for Trial train_tune_pbt_c9d37_00006
    2023-01-09 15:50:16,584	INFO pbt.py:646 -- [pbt]: no checkpoint for trial. Skip exploit for Trial train_tune_pbt_c9d37_00004
    2023-01-09 15:50:24,126	INFO pbt.py:646 -- [pbt]: no checkpoint for trial. Skip exploit for Trial train_tune_pbt_c9d37_00007
    2023-01-09 15:52:37,178	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial c9d37_00000 (score = 85.616667) into trial c9d37_00004 (score = 83.025000)
    
    2023-01-09 15:52:37,180	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trialc9d37_00004:
    lr : 0.04022534613227904 --- (* 0.8) --> 0.03218027690582323
    weight_decay : 0.003839114213595841 --- (resample) --> 0.0027402907213869244
    
    2023-01-09 15:52:39,169	WARNING util.py:244 -- The `start_trial` operation took 0.989 s, which may be a performance bottleneck.
    2023-01-09 15:53:44,019	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial c9d37_00000 (score = 86.100000) into trial c9d37_00001 (score = 81.916667)
    
    2023-01-09 15:53:44,020	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trialc9d37_00001:
    lr : 0.04022534613227904 --- (* 0.8) --> 0.03218027690582323
    weight_decay : 0.003839114213595841 --- (* 1.2) --> 0.004606937056315009
    
    2023-01-09 15:53:45,962	WARNING util.py:244 -- The `start_trial` operation took 0.754 s, which may be a performance bottleneck.
    2023-01-09 15:54:52,608	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial c9d37_00004 (score = 86.191667) into trial c9d37_00000 (score = 85.133333)
    
    2023-01-09 15:54:52,610	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trialc9d37_00000:
    lr : 0.03218027690582323 --- (* 1.2) --> 0.038616332286987874
    weight_decay : 0.0027402907213869244 --- (* 0.8) --> 0.0021922325771095395
    
    2023-01-09 15:54:54,019	WARNING util.py:244 -- The `start_trial` operation took 0.783 s, which may be a performance bottleneck.
    2023-01-09 15:55:09,443	INFO pbt.py:646 -- [pbt]: no checkpoint for trial. Skip exploit for Trial train_tune_pbt_c9d37_00004
    2023-01-09 15:55:33,685	INFO tune.py:762 -- Total run time: 378.58 seconds (377.81 seconds for the tuning loop).



```python
print(results_pbt.get_best_result())

ax = None
dfs = {result.log_dir: result.metrics_dataframe for result in results_pbt}
for d in dfs.values():
    ax = d.plot(ax=ax, y="mean_accuracy", x="epoch", legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean accuracy")
```

    Result(metrics={'mean_accuracy': 86.75, 'epoch': 10, 'should_checkpoint': True, 'done': True, 'trial_id': 'c9d37_00000', 'experiment_tag': '0_lr=0.0402,weight_decay=0.0038@perturbed[lr=0.0386,weight_decay=0.0022]'}, error=None, log_dir=PosixPath('/home/raffi/ray_results/train_tune_pbt_2023-01-09_15-49-15/train_tune_pbt_c9d37_00000_0_lr=0.0402,weight_decay=0.0038_2023-01-09_15-49-20'))





    Text(0, 0.5, 'Mean accuracy')




    
![png](Ray_Tune_FashionMNIST_files/Ray_Tune_FashionMNIST_41_2.png)
    



```python
!jupyter nbconvert --to markdown Ray_Tune_FashionMNIST.ipynb
```

    [NbConvertApp] Converting notebook Ray_Tune_FashionMNIST.ipynb to markdown
    [NbConvertApp] Support files will be in Ray_Tune_FashionMNIST_files/
    [NbConvertApp] Making directory Ray_Tune_FashionMNIST_files
    [NbConvertApp] Making directory Ray_Tune_FashionMNIST_files
    [NbConvertApp] Making directory Ray_Tune_FashionMNIST_files
    [NbConvertApp] Writing 64503 bytes to Ray_Tune_FashionMNIST.md

