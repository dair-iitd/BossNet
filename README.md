# BossNet
BossNet: Disentangling Language and Knowledge in Task Oriented Dialogs

## Datasets
We make use of the standard bAbI datasets which can be found here.
Additionally we make use of CamRest676 which can be found here.
Finally we use SMD which can be found here.

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
- `--task_id` this is task dependent. 1-5 for bAbI, 6 for CamRest, and 7 for SMD
- `--embedding_size` hidden state size of the two rnn
- `--batch_size` batch size
- `--learning_rate` learning rate
- `--word_drop_prob` dropout rate
- `--hops` number of stacked rnn layers, or number of hops for BossNet

Look at `params.py` for detailed information on the runtime options

## Testing
To test the best model and get accuracies run `single_dialog.py` with --train=False.
