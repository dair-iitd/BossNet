# BossNet
BossNet: Disentangling Language and Knowledge in Task Oriented Dialogs

## Datasets
The complete data folder structure with all tasks can be downloaded [here](https://drive.google.com/open?id=11D-ui5KiQHQN45yVc2PqthxO3tWJH4vW).  

### Individual Datasets
- [bAbI Dialog](https://drive.google.com/open?id=1fc1CKlJi_DJcr0kGrwwa2x4OiZyinavN) (26.2 MB)
- [CamRest](https://drive.google.com/open?id=1TIo74qjRiGeZNOiLKWit72a98lgv75MN) (1.1 MB)
- [Stanford Multi Domain Dataset / SMD](https://drive.google.com/open?id=1KRYx9HgpSeNkdNCzyzUiUzR_zmt0dyh7) (6.7 MB)

## Run Environment
We include a `requirements.txt` which has all the libraries installed for the correct run of the BossNet code.
For best practices create a virtual / conda environment and install all dependencies via:
```console
❱❱❱ pip install -r requirements.txt
```

## Training
The model is run using the script `main.py` 
```console
❱❱❱ python main.py --task_id 1 --train 
```

The list of parameters to run the script is:
- `--task_id` this is task dependent. 1-5 for bAbI, 7 for CamRest, and 8 for SMD
- `--batch_size` batch size
- `--learning_rate` learning rate
- `--embedding_size` hidden state size of the two rnn
- `--hops` number of stacked rnn layers for BossNet
- `--word_drop_prob` dropout rate
- `--p_gen_loss_weight` loss function weight on copy

Look at `params.py` for detailed information on the runtime options

### Training from Saved Model
There is support to start training from a previously saved checkpoint with the *--save* flag.

## Testing
To obtain metric scores on the best model run `main.py` with *--train=False* or by omitting the train flag altogether. Make sure all the parameter options match those of the trained model.
```console
❱❱❱ python main.py --task_id 1
```

# Analysis

## Trainable Paramters
The following numbers are reported using *embedding_size 256*, *batch_size 64*, *hops 6*  

|        | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | CamRest | SMD |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **BossNet** | 1204993 | 1198849 | 1478913 | 1715457 | 1943809 | 1968129 | 2187265 |
| **Mem2Seq** | 776030 | 780127 | 3426789 | 4508397 | 5049201 | 5274536 | 6880560 |
| **Seq2Seq + Copy** | 6379859 | 6378321 | 6870481 | 7083494 | 7167315 | 7223452 | 7504137 |
| **Seq2Seq** | 6905172 | 6903634 | 7395794 | 7608807 | 7692628 | 7748765 | 8029450 |

## Running Times
The following numbers are reported using *embedding_size 128*, *batch_size 64*, *hops 3*  
Times are reported as ` sec. per train epoch (avg. no. of epochs till convergence) **total train time** `

|        | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | CamRest | SMD |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **BossNet** | 38.1 (15) **571.5** | 65.2 (10) **652.0** | 142.4 (25) **3560** | 16.9 (2) **33.8** | 231.3 (6) **1387.8** | 113.5 (6) **681** | 1252 (10) **12520** |
| **Mem2Seq** | 10 (100) **1000** | 32 (30) **960** | 51 (90) **4590** | 4 (10) **40** | 136 (60) **8160** | 22 (40) **880** | 81 (40) **3240** |

## Hyperparameters

|        | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | CamRest | SMD |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Learning Rate | 0.001 | 0.001 | 0.005 | 0.001 | 0.0005 | 0.0005 | 0.0005 |
| Hops | 1 | 1 | 3 | 1 | 3 | 6 | 3 |
| Embedding Size | 128 | 128 | 128 | 128 | 256 | 256 | 256 |
| Disentangle Loss Weight | 1.0 | 1.0 | 1.5 | 1.0 | 1.0 | 1.0 | 1.0 |
| DLD | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.1 |
