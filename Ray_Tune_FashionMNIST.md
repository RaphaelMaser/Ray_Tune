[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RaphaelMaser/Ray_Tune/blob/main/Ray_Tune_FashionMNIST.ipynb)

## Ray

<img src="./images/ray_header_logo.png" 
     align="middle" 
     width="1000" 
     title="Ray Logo (https://docs.ray.io/en/latest/index.html)"/>

*Ray Logo (https://docs.ray.io/en/latest/index.html)*

Ray is a framework for distributed computing in Python. The core idea of Ray is to make distributed computing as accessible as possible while not disturbing the workflow of the user. For that purpose it integrates well with all of the mainstream Python libraries used in distributed scenarios like PyTorch, Tensorflow, Scikit-learn, etc.. Integrating Ray in your own workflow typically only needs small changes in your code, there is no need to re-write your application.

In general Ray consists of several different libraries:

<img src="./images/what-is-ray-padded.svg" 
     align="center" 
     width="1200" />

*Ray Framework Overview (https://docs.ray.io/en/latest/index.html)*

Ray Core provides the capability to build clusters and distribute any workload across the nodes of the cluster. Whether that be your PC and laptop or some gpu nodes of the iris-HPC. Ray schedules the tasks across the nodes and manages data transfers between the nodes.

Ray Core is a low-level library and although it is well made it needs some knowledge to use it for more sophisticated tasks like hyperparameter optimization (HPO) or NN training. Ray's higher-level libraries like Ray Train (NN training) and Ray Tune (HPO) use the Ray Core library to provide a simple interface for interacting with Ray without needing to know the details of distributed training.

With Ray Tune only a few lines of code needs to be added to the training procedure to run a full HPO with state-of-the-art search algorithms and schedulers.

### Why should I use Ray Tune and not any other HPO library?

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
     Accuracy: 16.2%, Avg loss: 2.290474 
    
    Epoch 2
    -------------------------------
    Test Error: 
     Accuracy: 21.9%, Avg loss: 2.278679 
    
    Epoch 3
    -------------------------------
    Test Error: 
     Accuracy: 27.2%, Avg loss: 2.267174 
    
    Epoch 4
    -------------------------------
    Test Error: 
     Accuracy: 31.9%, Avg loss: 2.255836 
    
    Epoch 5
    -------------------------------
    Test Error: 
     Accuracy: 35.0%, Avg loss: 2.244509 
    
    Epoch 6
    -------------------------------
    Test Error: 
     Accuracy: 37.1%, Avg loss: 2.233042 
    
    Epoch 7
    -------------------------------
    Test Error: 
     Accuracy: 38.4%, Avg loss: 2.221304 
    
    Epoch 8
    -------------------------------
    Test Error: 
     Accuracy: 39.7%, Avg loss: 2.209186 
    
    Epoch 9
    -------------------------------
    Test Error: 
     Accuracy: 40.8%, Avg loss: 2.196593 
    
    Epoch 10
    -------------------------------
    Test Error: 
     Accuracy: 42.1%, Avg loss: 2.183453 
    


The final accuracy of the model after 10 epochs is ~42 %.

## Hyperparameter optimization (Random Search)

Now we are going to use Ray to do a small random hyperparameter search. 

The following line is not mandatory, normally Ray initializes itself if it was not initialized before, but in this case I want to set the log_to_driver argument to "False" to suppress the output of the trials. Otherwise the output of the different trails will be mixed in no special order because of the concurrent execution.


```python
ray.init(log_to_driver=False, ignore_reinit_error=True)
```

    2023-01-07 17:48:50,030	INFO worker.py:1538 -- Started a local Ray instance.





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
        session.report(metrics={"mean_accuracy": accuracy})
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


```python
print(results.get_best_result())

ax = None
dfs = {result.log_dir: result.metrics_dataframe for result in results}
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean accuracy")
```

    Result(metrics={'mean_accuracy': 87.51666666666667, 'done': True, 'trial_id': '0ff61_00002', 'experiment_tag': '2_lr=0.0509,weight_decay=0.0001'}, error=None, log_dir=PosixPath('/home/raffi/ray_results/train_tune_2023-01-07_17-05-09/train_tune_0ff61_00002_2_lr=0.0509,weight_decay=0.0001_2023-01-07_17-05-19'))





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

results = tuner.fit()
```


<div class="tuneStatus">
  <div style="display: flex;flex-direction: row">
    <div style="display: flex;flex-direction: column;">
      <h3>Tune Status</h3>
      <table>
<tbody>
<tr><td>Current time:</td><td>2023-01-07 17:27:06</td></tr>
<tr><td>Running for: </td><td>00:03:11.40        </td></tr>
<tr><td>Memory:      </td><td>15.8/30.6 GiB      </td></tr>
</tbody>
</table>
    </div>
    <div class="vDivider"></div>
    <div class="systemInfo">
      <h3>System Info</h3>
      Using AsyncHyperBand: num_stopped=16<br>Bracket: Iter 4.000: 85.53125 | Iter 1.000: 81.05833333333334<br>Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/10.47 GiB heap, 0.0/5.24 GiB objects
    </div>

  </div>
  <div class="hDivider"></div>
  <div class="trialStatus">
    <h3>Trial Status</h3>
    <table>
<thead>
<tr><th>Trial name            </th><th>status    </th><th>loc                  </th><th style="text-align: right;">       lr</th><th style="text-align: right;">  weight_decay</th><th style="text-align: right;">    acc</th><th style="text-align: right;">  iter</th><th style="text-align: right;">  total time (s)</th></tr>
</thead>
<tbody>
<tr><td>train_tune_adf3c_00000</td><td>TERMINATED</td><td>192.168.188.20:788216</td><td style="text-align: right;">0.0644557</td><td style="text-align: right;">   0.0040345  </td><td style="text-align: right;">86.5917</td><td style="text-align: right;">    10</td><td style="text-align: right;">        161.764 </td></tr>
<tr><td>train_tune_adf3c_00001</td><td>TERMINATED</td><td>192.168.188.20:788291</td><td style="text-align: right;">0.075588 </td><td style="text-align: right;">   0.00356234 </td><td style="text-align: right;">81.0417</td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.3533</td></tr>
<tr><td>train_tune_adf3c_00002</td><td>TERMINATED</td><td>192.168.188.20:788308</td><td style="text-align: right;">0.0550501</td><td style="text-align: right;">   0.00700203 </td><td style="text-align: right;">80.5583</td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.2717</td></tr>
<tr><td>train_tune_adf3c_00003</td><td>TERMINATED</td><td>192.168.188.20:788340</td><td style="text-align: right;">0.0238723</td><td style="text-align: right;">   0.00376137 </td><td style="text-align: right;">77.1167</td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.3274</td></tr>
<tr><td>train_tune_adf3c_00004</td><td>TERMINATED</td><td>192.168.188.20:788291</td><td style="text-align: right;">0.0864238</td><td style="text-align: right;">   0.00132724 </td><td style="text-align: right;">79.725 </td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.1607</td></tr>
<tr><td>train_tune_adf3c_00005</td><td>TERMINATED</td><td>192.168.188.20:788308</td><td style="text-align: right;">0.0152888</td><td style="text-align: right;">   0.00534216 </td><td style="text-align: right;">72.125 </td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.1705</td></tr>
<tr><td>train_tune_adf3c_00006</td><td>TERMINATED</td><td>192.168.188.20:788340</td><td style="text-align: right;">0.0884417</td><td style="text-align: right;">   0.000112171</td><td style="text-align: right;">88.025 </td><td style="text-align: right;">    10</td><td style="text-align: right;">        158.676 </td></tr>
<tr><td>train_tune_adf3c_00007</td><td>TERMINATED</td><td>192.168.188.20:788291</td><td style="text-align: right;">0.0391692</td><td style="text-align: right;">   0.00133369 </td><td style="text-align: right;">84.3333</td><td style="text-align: right;">     4</td><td style="text-align: right;">         64.9613</td></tr>
<tr><td>train_tune_adf3c_00008</td><td>TERMINATED</td><td>192.168.188.20:788308</td><td style="text-align: right;">0.0403739</td><td style="text-align: right;">   0.00694623 </td><td style="text-align: right;">80.8083</td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.1696</td></tr>
<tr><td>train_tune_adf3c_00009</td><td>TERMINATED</td><td>192.168.188.20:788308</td><td style="text-align: right;">0.0850602</td><td style="text-align: right;">   0.00863932 </td><td style="text-align: right;">80.6417</td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.3448</td></tr>
<tr><td>train_tune_adf3c_00010</td><td>TERMINATED</td><td>192.168.188.20:788308</td><td style="text-align: right;">0.0353943</td><td style="text-align: right;">   0.00581399 </td><td style="text-align: right;">79.075 </td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.1625</td></tr>
<tr><td>train_tune_adf3c_00011</td><td>TERMINATED</td><td>192.168.188.20:788308</td><td style="text-align: right;">0.0796056</td><td style="text-align: right;">   0.00203633 </td><td style="text-align: right;">85.1583</td><td style="text-align: right;">     4</td><td style="text-align: right;">         65.412 </td></tr>
<tr><td>train_tune_adf3c_00012</td><td>TERMINATED</td><td>192.168.188.20:788291</td><td style="text-align: right;">0.0653822</td><td style="text-align: right;">   0.00124497 </td><td style="text-align: right;">79.325 </td><td style="text-align: right;">     1</td><td style="text-align: right;">         17.0226</td></tr>
<tr><td>train_tune_adf3c_00013</td><td>TERMINATED</td><td>192.168.188.20:788291</td><td style="text-align: right;">0.0347586</td><td style="text-align: right;">   0.0081263  </td><td style="text-align: right;">76.6   </td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.2734</td></tr>
<tr><td>train_tune_adf3c_00014</td><td>TERMINATED</td><td>192.168.188.20:788291</td><td style="text-align: right;">0.0694535</td><td style="text-align: right;">   0.00875965 </td><td style="text-align: right;">80.8917</td><td style="text-align: right;">     1</td><td style="text-align: right;">         16.3038</td></tr>
<tr><td>train_tune_adf3c_00015</td><td>TERMINATED</td><td>192.168.188.20:788308</td><td style="text-align: right;">0.0698921</td><td style="text-align: right;">   0.00173578 </td><td style="text-align: right;">81     </td><td style="text-align: right;">     1</td><td style="text-align: right;">         14.6412</td></tr>
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



    2023-01-07 17:24:04,615	WARNING worker.py:1851 -- Warning: The actor ImplicitFunc is very large (45 MiB). Check that its definition is not implicitly capturing a large array or other object in scope. Tip: use ray.put() to put large objects in the Ray object store.
    2023-01-07 17:24:04,785	WARNING util.py:244 -- The `start_trial` operation took 1.007 s, which may be a performance bottleneck.



<div class="trialProgress">
  <h3>Trial Progress</h3>
  <table>
<thead>
<tr><th>Trial name            </th><th>date               </th><th>done  </th><th>episodes_total  </th><th>experiment_id                   </th><th>hostname  </th><th style="text-align: right;">  iterations_since_restore</th><th style="text-align: right;">  mean_accuracy</th><th>node_ip       </th><th style="text-align: right;">   pid</th><th style="text-align: right;">  time_since_restore</th><th style="text-align: right;">  time_this_iter_s</th><th style="text-align: right;">  time_total_s</th><th style="text-align: right;">  timestamp</th><th style="text-align: right;">  timesteps_since_restore</th><th>timesteps_total  </th><th style="text-align: right;">  training_iteration</th><th>trial_id   </th><th style="text-align: right;">  warmup_time</th></tr>
</thead>
<tbody>
<tr><td>train_tune_adf3c_00000</td><td>2023-01-07_17-26-48</td><td>True  </td><td>                </td><td>b13cb9eb0c1f46039afa460d394d168f</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        86.5917</td><td>192.168.188.20</td><td style="text-align: right;">788216</td><td style="text-align: right;">            161.764 </td><td style="text-align: right;">           15.376 </td><td style="text-align: right;">      161.764 </td><td style="text-align: right;"> 1673108808</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>adf3c_00000</td><td style="text-align: right;">   0.00383306</td></tr>
<tr><td>train_tune_adf3c_00001</td><td>2023-01-07_17-24-26</td><td>True  </td><td>                </td><td>8c67c849115d46eab6456da3da4569fb</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        81.0417</td><td>192.168.188.20</td><td style="text-align: right;">788291</td><td style="text-align: right;">             16.3533</td><td style="text-align: right;">           16.3533</td><td style="text-align: right;">       16.3533</td><td style="text-align: right;"> 1673108666</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00001</td><td style="text-align: right;">   0.00540042</td></tr>
<tr><td>train_tune_adf3c_00002</td><td>2023-01-07_17-24-27</td><td>True  </td><td>                </td><td>5ad4e8296b7e4ef599efb228e817a2ef</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        80.5583</td><td>192.168.188.20</td><td style="text-align: right;">788308</td><td style="text-align: right;">             16.2717</td><td style="text-align: right;">           16.2717</td><td style="text-align: right;">       16.2717</td><td style="text-align: right;"> 1673108667</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00002</td><td style="text-align: right;">   0.00467658</td></tr>
<tr><td>train_tune_adf3c_00003</td><td>2023-01-07_17-24-27</td><td>True  </td><td>                </td><td>e490fcff9b3b40e2bd0374683ecd728c</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        77.1167</td><td>192.168.188.20</td><td style="text-align: right;">788340</td><td style="text-align: right;">             16.3274</td><td style="text-align: right;">           16.3274</td><td style="text-align: right;">       16.3274</td><td style="text-align: right;"> 1673108667</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00003</td><td style="text-align: right;">   0.00535107</td></tr>
<tr><td>train_tune_adf3c_00004</td><td>2023-01-07_17-24-42</td><td>True  </td><td>                </td><td>8c67c849115d46eab6456da3da4569fb</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        79.725 </td><td>192.168.188.20</td><td style="text-align: right;">788291</td><td style="text-align: right;">             16.1607</td><td style="text-align: right;">           16.1607</td><td style="text-align: right;">       16.1607</td><td style="text-align: right;"> 1673108682</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00004</td><td style="text-align: right;">   0.00540042</td></tr>
<tr><td>train_tune_adf3c_00005</td><td>2023-01-07_17-24-43</td><td>True  </td><td>                </td><td>5ad4e8296b7e4ef599efb228e817a2ef</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        72.125 </td><td>192.168.188.20</td><td style="text-align: right;">788308</td><td style="text-align: right;">             16.1705</td><td style="text-align: right;">           16.1705</td><td style="text-align: right;">       16.1705</td><td style="text-align: right;"> 1673108683</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00005</td><td style="text-align: right;">   0.00467658</td></tr>
<tr><td>train_tune_adf3c_00006</td><td>2023-01-07_17-27-06</td><td>True  </td><td>                </td><td>e490fcff9b3b40e2bd0374683ecd728c</td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        88.025 </td><td>192.168.188.20</td><td style="text-align: right;">788340</td><td style="text-align: right;">            158.676 </td><td style="text-align: right;">           12.4835</td><td style="text-align: right;">      158.676 </td><td style="text-align: right;"> 1673108826</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>adf3c_00006</td><td style="text-align: right;">   0.00535107</td></tr>
<tr><td>train_tune_adf3c_00007</td><td>2023-01-07_17-25-47</td><td>True  </td><td>                </td><td>8c67c849115d46eab6456da3da4569fb</td><td>fedora    </td><td style="text-align: right;">                         4</td><td style="text-align: right;">        84.3333</td><td>192.168.188.20</td><td style="text-align: right;">788291</td><td style="text-align: right;">             64.9613</td><td style="text-align: right;">           16.3468</td><td style="text-align: right;">       64.9613</td><td style="text-align: right;"> 1673108747</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   4</td><td>adf3c_00007</td><td style="text-align: right;">   0.00540042</td></tr>
<tr><td>train_tune_adf3c_00008</td><td>2023-01-07_17-24-59</td><td>True  </td><td>                </td><td>5ad4e8296b7e4ef599efb228e817a2ef</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        80.8083</td><td>192.168.188.20</td><td style="text-align: right;">788308</td><td style="text-align: right;">             16.1696</td><td style="text-align: right;">           16.1696</td><td style="text-align: right;">       16.1696</td><td style="text-align: right;"> 1673108699</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00008</td><td style="text-align: right;">   0.00467658</td></tr>
<tr><td>train_tune_adf3c_00009</td><td>2023-01-07_17-25-15</td><td>True  </td><td>                </td><td>5ad4e8296b7e4ef599efb228e817a2ef</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        80.6417</td><td>192.168.188.20</td><td style="text-align: right;">788308</td><td style="text-align: right;">             16.3448</td><td style="text-align: right;">           16.3448</td><td style="text-align: right;">       16.3448</td><td style="text-align: right;"> 1673108715</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00009</td><td style="text-align: right;">   0.00467658</td></tr>
<tr><td>train_tune_adf3c_00010</td><td>2023-01-07_17-25-31</td><td>True  </td><td>                </td><td>5ad4e8296b7e4ef599efb228e817a2ef</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        79.075 </td><td>192.168.188.20</td><td style="text-align: right;">788308</td><td style="text-align: right;">             16.1625</td><td style="text-align: right;">           16.1625</td><td style="text-align: right;">       16.1625</td><td style="text-align: right;"> 1673108731</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00010</td><td style="text-align: right;">   0.00467658</td></tr>
<tr><td>train_tune_adf3c_00011</td><td>2023-01-07_17-26-37</td><td>True  </td><td>                </td><td>5ad4e8296b7e4ef599efb228e817a2ef</td><td>fedora    </td><td style="text-align: right;">                         4</td><td style="text-align: right;">        85.1583</td><td>192.168.188.20</td><td style="text-align: right;">788308</td><td style="text-align: right;">             65.412 </td><td style="text-align: right;">           16.2313</td><td style="text-align: right;">       65.412 </td><td style="text-align: right;"> 1673108797</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   4</td><td>adf3c_00011</td><td style="text-align: right;">   0.00467658</td></tr>
<tr><td>train_tune_adf3c_00012</td><td>2023-01-07_17-26-05</td><td>True  </td><td>                </td><td>8c67c849115d46eab6456da3da4569fb</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        79.325 </td><td>192.168.188.20</td><td style="text-align: right;">788291</td><td style="text-align: right;">             17.0226</td><td style="text-align: right;">           17.0226</td><td style="text-align: right;">       17.0226</td><td style="text-align: right;"> 1673108765</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00012</td><td style="text-align: right;">   0.00540042</td></tr>
<tr><td>train_tune_adf3c_00013</td><td>2023-01-07_17-26-21</td><td>True  </td><td>                </td><td>8c67c849115d46eab6456da3da4569fb</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        76.6   </td><td>192.168.188.20</td><td style="text-align: right;">788291</td><td style="text-align: right;">             16.2734</td><td style="text-align: right;">           16.2734</td><td style="text-align: right;">       16.2734</td><td style="text-align: right;"> 1673108781</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00013</td><td style="text-align: right;">   0.00540042</td></tr>
<tr><td>train_tune_adf3c_00014</td><td>2023-01-07_17-26-37</td><td>True  </td><td>                </td><td>8c67c849115d46eab6456da3da4569fb</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        80.8917</td><td>192.168.188.20</td><td style="text-align: right;">788291</td><td style="text-align: right;">             16.3038</td><td style="text-align: right;">           16.3038</td><td style="text-align: right;">       16.3038</td><td style="text-align: right;"> 1673108797</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00014</td><td style="text-align: right;">   0.00540042</td></tr>
<tr><td>train_tune_adf3c_00015</td><td>2023-01-07_17-26-52</td><td>True  </td><td>                </td><td>5ad4e8296b7e4ef599efb228e817a2ef</td><td>fedora    </td><td style="text-align: right;">                         1</td><td style="text-align: right;">        81     </td><td>192.168.188.20</td><td style="text-align: right;">788308</td><td style="text-align: right;">             14.6412</td><td style="text-align: right;">           14.6412</td><td style="text-align: right;">       14.6412</td><td style="text-align: right;"> 1673108812</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                   1</td><td>adf3c_00015</td><td style="text-align: right;">   0.00467658</td></tr>
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



    2023-01-07 17:27:06,370	INFO tune.py:762 -- Total run time: 192.20 seconds (191.39 seconds for the tuning loop).



```python
print(results.get_best_result())

ax = None
dfs = {result.log_dir: result.metrics_dataframe for result in results}
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean accuracy")
```

    Result(metrics={'mean_accuracy': 88.02499999999999, 'done': True, 'trial_id': 'adf3c_00006', 'experiment_tag': '6_lr=0.0884,weight_decay=0.0001'}, error=None, log_dir=PosixPath('/home/raffi/ray_results/train_tune_2023-01-07_17-23-54/train_tune_adf3c_00006_6_lr=0.0884,weight_decay=0.0001_2023-01-07_17-24-27'))





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
        session.report(metrics={"mean_accuracy": accuracy}, checkpoint=checkpoint)
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
results = tuner.fit()
```


<div class="tuneStatus">
  <div style="display: flex;flex-direction: row">
    <div style="display: flex;flex-direction: column;">
      <h3>Tune Status</h3>
      <table>
<tbody>
<tr><td>Current time:</td><td>2023-01-07 18:14:09</td></tr>
<tr><td>Running for: </td><td>00:04:46.61        </td></tr>
<tr><td>Memory:      </td><td>15.8/30.6 GiB      </td></tr>
</tbody>
</table>
    </div>
    <div class="vDivider"></div>
    <div class="systemInfo">
      <h3>System Info</h3>
      PopulationBasedTraining: 14 checkpoints, 8 perturbs<br>Resources requested: 0/16 CPUs, 0/0 GPUs, 0.0/10.29 GiB heap, 0.0/5.14 GiB objects
    </div>

  </div>
  <div class="hDivider"></div>
  <div class="trialStatus">
    <h3>Trial Status</h3>
    <table>
<thead>
<tr><th>Trial name                </th><th>status    </th><th>loc                  </th><th style="text-align: right;">       lr</th><th style="text-align: right;">  weight_decay</th><th style="text-align: right;">    acc</th><th style="text-align: right;">  iter</th><th style="text-align: right;">  total time (s)</th></tr>
</thead>
<tbody>
<tr><td>train_tune_pbt_07f60_00000</td><td>TERMINATED</td><td>192.168.188.20:804711</td><td style="text-align: right;">0.0828772</td><td style="text-align: right;">   0.00158063 </td><td style="text-align: right;">86.3667</td><td style="text-align: right;">    10</td><td style="text-align: right;">         228.357</td></tr>
<tr><td>train_tune_pbt_07f60_00001</td><td>TERMINATED</td><td>192.168.188.20:804808</td><td style="text-align: right;">0.0691859</td><td style="text-align: right;">   5.30056e-05</td><td style="text-align: right;">86.775 </td><td style="text-align: right;">    10</td><td style="text-align: right;">         230.84 </td></tr>
<tr><td>train_tune_pbt_07f60_00002</td><td>TERMINATED</td><td>192.168.188.20:805911</td><td style="text-align: right;">0.0553487</td><td style="text-align: right;">   0.000813207</td><td style="text-align: right;">86.4917</td><td style="text-align: right;">    10</td><td style="text-align: right;">         227.828</td></tr>
<tr><td>train_tune_pbt_07f60_00003</td><td>TERMINATED</td><td>192.168.188.20:806220</td><td style="text-align: right;">0.0225328</td><td style="text-align: right;">   0.00189675 </td><td style="text-align: right;">88.05  </td><td style="text-align: right;">    10</td><td style="text-align: right;">         225.021</td></tr>
<tr><td>train_tune_pbt_07f60_00004</td><td>TERMINATED</td><td>192.168.188.20:805004</td><td style="text-align: right;">0.0442789</td><td style="text-align: right;">   0.000650565</td><td style="text-align: right;">87.675 </td><td style="text-align: right;">    10</td><td style="text-align: right;">         213.853</td></tr>
<tr><td>train_tune_pbt_07f60_00005</td><td>TERMINATED</td><td>192.168.188.20:806201</td><td style="text-align: right;">0.083023 </td><td style="text-align: right;">   6.36067e-05</td><td style="text-align: right;">87.45  </td><td style="text-align: right;">    10</td><td style="text-align: right;">         227.114</td></tr>
<tr><td>train_tune_pbt_07f60_00006</td><td>TERMINATED</td><td>192.168.188.20:805429</td><td style="text-align: right;">0.0064898</td><td style="text-align: right;">   6.36067e-05</td><td style="text-align: right;">86.4583</td><td style="text-align: right;">    10</td><td style="text-align: right;">         230.921</td></tr>
<tr><td>train_tune_pbt_07f60_00007</td><td>TERMINATED</td><td>192.168.188.20:806539</td><td style="text-align: right;">0.0354232</td><td style="text-align: right;">   0.00799822 </td><td style="text-align: right;">86.9167</td><td style="text-align: right;">    10</td><td style="text-align: right;">         209.585</td></tr>
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



    2023-01-07 18:09:28,603	WARNING worker.py:1851 -- Warning: The actor ImplicitFunc is very large (45 MiB). Check that its definition is not implicitly capturing a large array or other object in scope. Tip: use ray.put() to put large objects in the Ray object store.
    2023-01-07 18:09:28,800	WARNING util.py:244 -- The `start_trial` operation took 1.062 s, which may be a performance bottleneck.
    2023-01-07 18:09:32,470	WARNING util.py:244 -- The `start_trial` operation took 0.512 s, which may be a performance bottleneck.
    2023-01-07 18:09:34,001	WARNING util.py:244 -- The `start_trial` operation took 0.560 s, which may be a performance bottleneck.
    2023-01-07 18:09:38,305	WARNING util.py:244 -- The `start_trial` operation took 0.583 s, which may be a performance bottleneck.



<div class="trialProgress">
  <h3>Trial Progress</h3>
  <table>
<thead>
<tr><th>Trial name                </th><th>date               </th><th>done  </th><th>episodes_total  </th><th>experiment_id                   </th><th>experiment_tag                                                          </th><th>hostname  </th><th style="text-align: right;">  iterations_since_restore</th><th style="text-align: right;">  mean_accuracy</th><th>node_ip       </th><th style="text-align: right;">   pid</th><th>should_checkpoint  </th><th style="text-align: right;">  time_since_restore</th><th style="text-align: right;">  time_this_iter_s</th><th style="text-align: right;">  time_total_s</th><th style="text-align: right;">  timestamp</th><th style="text-align: right;">  timesteps_since_restore</th><th>timesteps_total  </th><th style="text-align: right;">  training_iteration</th><th>trial_id   </th><th style="text-align: right;">  warmup_time</th></tr>
</thead>
<tbody>
<tr><td>train_tune_pbt_07f60_00000</td><td>2023-01-07_18-13-19</td><td>True  </td><td>                </td><td>ddf27a2fa1e243848f7bcf8f03ff8d2b</td><td>0_lr=0.0829,weight_decay=0.0016                                         </td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        86.3667</td><td>192.168.188.20</td><td style="text-align: right;">804711</td><td>True               </td><td style="text-align: right;">            228.357 </td><td style="text-align: right;">           22.6814</td><td style="text-align: right;">       228.357</td><td style="text-align: right;"> 1673111599</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>07f60_00000</td><td style="text-align: right;">   0.00388479</td></tr>
<tr><td>train_tune_pbt_07f60_00001</td><td>2023-01-07_18-13-26</td><td>True  </td><td>                </td><td>622663ab4ea64532af95891e08935e6f</td><td>1_lr=0.0692,weight_decay=0.0001                                         </td><td>fedora    </td><td style="text-align: right;">                        10</td><td style="text-align: right;">        86.775 </td><td>192.168.188.20</td><td style="text-align: right;">804808</td><td>True               </td><td style="text-align: right;">            230.84  </td><td style="text-align: right;">           23.0601</td><td style="text-align: right;">       230.84 </td><td style="text-align: right;"> 1673111606</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>07f60_00001</td><td style="text-align: right;">   0.00565958</td></tr>
<tr><td>train_tune_pbt_07f60_00002</td><td>2023-01-07_18-13-33</td><td>True  </td><td>                </td><td>622663ab4ea64532af95891e08935e6f</td><td>2_lr=0.0720,weight_decay=0.0088@perturbed[lr=0.0553,weight_decay=0.0008]</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        86.4917</td><td>192.168.188.20</td><td style="text-align: right;">805911</td><td>True               </td><td style="text-align: right;">             44.0929</td><td style="text-align: right;">           21.2517</td><td style="text-align: right;">       227.828</td><td style="text-align: right;"> 1673111613</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>07f60_00002</td><td style="text-align: right;">   0.0226119 </td></tr>
<tr><td>train_tune_pbt_07f60_00003</td><td>2023-01-07_18-13-39</td><td>True  </td><td>                </td><td>ddf27a2fa1e243848f7bcf8f03ff8d2b</td><td>3_lr=0.0620,weight_decay=0.0026@perturbed[lr=0.0225,weight_decay=0.0019]</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        88.05  </td><td>192.168.188.20</td><td style="text-align: right;">806220</td><td>True               </td><td style="text-align: right;">             42.7712</td><td style="text-align: right;">           18.6593</td><td style="text-align: right;">       225.021</td><td style="text-align: right;"> 1673111619</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>07f60_00003</td><td style="text-align: right;">   0.0291586 </td></tr>
<tr><td>train_tune_pbt_07f60_00004</td><td>2023-01-07_18-13-57</td><td>True  </td><td>                </td><td>622663ab4ea64532af95891e08935e6f</td><td>4_lr=0.0351,weight_decay=0.0084@perturbed[lr=0.0443,weight_decay=0.0007]</td><td>fedora    </td><td style="text-align: right;">                         6</td><td style="text-align: right;">        87.675 </td><td>192.168.188.20</td><td style="text-align: right;">805004</td><td>True               </td><td style="text-align: right;">            122.264 </td><td style="text-align: right;">           13.1163</td><td style="text-align: right;">       213.853</td><td style="text-align: right;"> 1673111637</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>07f60_00004</td><td style="text-align: right;">   0.00573421</td></tr>
<tr><td>train_tune_pbt_07f60_00005</td><td>2023-01-07_18-13-38</td><td>True  </td><td>                </td><td>622663ab4ea64532af95891e08935e6f</td><td>5_lr=0.0956,weight_decay=0.0018@perturbed[lr=0.0830,weight_decay=0.0001]</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        87.45  </td><td>192.168.188.20</td><td style="text-align: right;">806201</td><td>True               </td><td style="text-align: right;">             42.7993</td><td style="text-align: right;">           19.2309</td><td style="text-align: right;">       227.114</td><td style="text-align: right;"> 1673111618</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>07f60_00005</td><td style="text-align: right;">   0.0242839 </td></tr>
<tr><td>train_tune_pbt_07f60_00006</td><td>2023-01-07_18-13-33</td><td>True  </td><td>                </td><td>622663ab4ea64532af95891e08935e6f</td><td>6_lr=0.0358,weight_decay=0.0035@perturbed[lr=0.0065,weight_decay=0.0001]</td><td>fedora    </td><td style="text-align: right;">                         8</td><td style="text-align: right;">        86.4583</td><td>192.168.188.20</td><td style="text-align: right;">805429</td><td>True               </td><td style="text-align: right;">            184.699 </td><td style="text-align: right;">           21.6369</td><td style="text-align: right;">       230.921</td><td style="text-align: right;"> 1673111613</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>07f60_00006</td><td style="text-align: right;">   0.026787  </td></tr>
<tr><td>train_tune_pbt_07f60_00007</td><td>2023-01-07_18-14-09</td><td>True  </td><td>                </td><td>622663ab4ea64532af95891e08935e6f</td><td>7_lr=0.0210,weight_decay=0.0049@perturbed[lr=0.0354,weight_decay=0.0080]</td><td>fedora    </td><td style="text-align: right;">                         2</td><td style="text-align: right;">        86.9167</td><td>192.168.188.20</td><td style="text-align: right;">806539</td><td>True               </td><td style="text-align: right;">             24.8776</td><td style="text-align: right;">           12.0098</td><td style="text-align: right;">       209.585</td><td style="text-align: right;"> 1673111649</td><td style="text-align: right;">                        0</td><td>                 </td><td style="text-align: right;">                  10</td><td>07f60_00007</td><td style="text-align: right;">   0.0147822 </td></tr>
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



    2023-01-07 18:10:22,683	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial 07f60_00001 (score = 84.858333) into trial 07f60_00006 (score = 82.366667)
    
    2023-01-07 18:10:22,685	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trial07f60_00006:
    lr : 0.06918585828577933 --- (resample) --> 0.006489804259717111
    weight_decay : 5.300560865403002e-05 --- (* 1.2) --> 6.360673038483603e-05
    
    2023-01-07 18:10:22,884	INFO pbt.py:646 -- [pbt]: no checkpoint for trial. Skip exploit for Trial train_tune_pbt_07f60_00004
    2023-01-07 18:10:23,838	WARNING util.py:244 -- The `start_trial` operation took 0.563 s, which may be a performance bottleneck.
    2023-01-07 18:10:23,840	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial 07f60_00001 (score = 84.858333) into trial 07f60_00002 (score = 81.066667)
    
    2023-01-07 18:10:23,842	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trial07f60_00002:
    lr : 0.06918585828577933 --- (* 0.8) --> 0.05534868662862347
    weight_decay : 5.300560865403002e-05 --- (resample) --> 0.0008132066046950878
    
    2023-01-07 18:10:24,913	WARNING util.py:244 -- The `start_trial` operation took 0.597 s, which may be a performance bottleneck.
    2023-01-07 18:10:29,204	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial 07f60_00001 (score = 84.858333) into trial 07f60_00007 (score = 80.000000)
    
    2023-01-07 18:10:29,205	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trial07f60_00007:
    lr : 0.06918585828577933 --- (* 1.2) --> 0.0830230299429352
    weight_decay : 5.300560865403002e-05 --- (* 0.8) --> 4.240448692322402e-05
    
    2023-01-07 18:10:30,802	WARNING util.py:244 -- The `start_trial` operation took 0.628 s, which may be a performance bottleneck.
    2023-01-07 18:11:54,604	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial 07f60_00002 (score = 86.408333) into trial 07f60_00004 (score = 83.908333)
    
    2023-01-07 18:11:54,605	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trial07f60_00004:
    lr : 0.05534868662862347 --- (* 0.8) --> 0.044278949302898774
    weight_decay : 0.0008132066046950878 --- (* 0.8) --> 0.0006505652837560703
    
    2023-01-07 18:11:55,894	WARNING util.py:244 -- The `start_trial` operation took 0.601 s, which may be a performance bottleneck.
    2023-01-07 18:11:57,842	WARNING util.py:244 -- The `start_trial` operation took 0.659 s, which may be a performance bottleneck.
    2023-01-07 18:12:48,904	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial 07f60_00001 (score = 88.358333) into trial 07f60_00005 (score = 86.025000)
    
    2023-01-07 18:12:48,906	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trial07f60_00005:
    lr : 0.06918585828577933 --- (* 1.2) --> 0.0830230299429352
    weight_decay : 5.300560865403002e-05 --- (* 1.2) --> 6.360673038483603e-05
    
    2023-01-07 18:12:50,780	WARNING util.py:244 -- The `start_trial` operation took 0.590 s, which may be a performance bottleneck.
    2023-01-07 18:12:50,783	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial 07f60_00000 (score = 87.708333) into trial 07f60_00003 (score = 85.175000)
    
    2023-01-07 18:12:50,785	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trial07f60_00003:
    lr : 0.08287719726491752 --- (resample) --> 0.02253278745418752
    weight_decay : 0.001580626729045409 --- (* 1.2) --> 0.0018967520748544907
    
    2023-01-07 18:12:51,802	WARNING util.py:244 -- The `start_trial` operation took 0.614 s, which may be a performance bottleneck.
    2023-01-07 18:12:52,317	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial 07f60_00000 (score = 87.708333) into trial 07f60_00007 (score = 85.875000)
    
    2023-01-07 18:12:52,318	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trial07f60_00007:
    lr : 0.08287719726491752 --- (* 1.2) --> 0.09945263671790101
    weight_decay : 0.001580626729045409 --- (* 1.2) --> 0.0018967520748544907
    
    2023-01-07 18:12:53,871	WARNING util.py:244 -- The `start_trial` operation took 0.679 s, which may be a performance bottleneck.
    2023-01-07 18:13:41,291	INFO pbt.py:804 -- 
    
    [PopulationBasedTraining] [Exploit] Cloning trial 07f60_00004 (score = 87.491667) into trial 07f60_00007 (score = 85.866667)
    
    2023-01-07 18:13:41,292	INFO pbt.py:831 -- 
    
    [PopulationBasedTraining] [Explore] Perturbed the hyperparameter config of trial07f60_00007:
    lr : 0.044278949302898774 --- (* 0.8) --> 0.03542315944231902
    weight_decay : 0.0006505652837560703 --- (resample) --> 0.007998216784259833
    
    2023-01-07 18:13:41,923	WARNING util.py:244 -- The `start_trial` operation took 0.506 s, which may be a performance bottleneck.
    2023-01-07 18:14:09,466	INFO tune.py:762 -- Total run time: 287.32 seconds (286.61 seconds for the tuning loop).



```python
print(results.get_best_result())
ax = None
dfs = {result.log_dir: result.metrics_dataframe for result in results}
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean accuracy")
```

    Result(metrics={'mean_accuracy': 88.05, 'should_checkpoint': True, 'done': True, 'trial_id': '07f60_00003', 'experiment_tag': '3_lr=0.0620,weight_decay=0.0026@perturbed[lr=0.0225,weight_decay=0.0019]'}, error=None, log_dir=PosixPath('/home/raffi/ray_results/train_tune_pbt_2023-01-07_18-09-22/train_tune_pbt_07f60_00003_3_lr=0.0620,weight_decay=0.0026_2023-01-07_18-09-31'))





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
    [NbConvertApp] Writing 60377 bytes to Ray_Tune_FashionMNIST.md

