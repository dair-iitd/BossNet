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
