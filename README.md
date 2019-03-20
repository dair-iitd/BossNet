# BossNet
BossNet: Disentangling Language and Knowledge in Task Oriented Dialogs

## Datasets
The complete data folder structure with all tasks can be downloaded here.  

### Individual Datasets
- bAbI Dialog (26.2 MB)
- CamRest (1.1 MB)
- Stanford Multi Domain Dataset / SMD (6.7 MB)

## Run Environment
We include a `requirements.txt` which has all the libraries installed for the correct run of the BossNet code.
For best practices create a virtual / conda environment and install all dependencies via:
```console
❱❱❱ pip install requirements.txt
```

## Training
The model is run using the script `main.py` 
```console
❱❱❱ python main.py --train --task_id 1 --batch_size 64 --word_drop_rate 0.2 --learning_rate 0.001 --embedding_size 128 --hops 3
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
To test the best model and get accuracies run `single_dialog.py` with *--train=False* or by omitting the train flag altogether.
```console
❱❱❱ python main.py --task_id 1 --batch_size 64 --word_drop_rate 0.2 --learning_rate 0.001 --embedding_size 128 --hops 3
```

## Models
You can access all the saved models for each task here. The parameters to run each model is mentioned in the paper.
