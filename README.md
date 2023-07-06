# EV-LayerSegNet: Self-supervised Motion Segmentation using Event Cameras

## Usage

This project uses Python >= 3.7.3 and we strongly recommend the use of virtual environments. If you don't have an environment manager yet, we recommend `pyenv`. It can be installed via:

```
curl https://pyenv.run | bash
```

Make sure your `~/.bashrc` file contains the following:

```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

After that, restart your terminal and run:

```
pyenv update
```

To set up your environment with `pyenv` first install the required python distribution and make sure the installation is successful (i.e., no errors nor warnings):

```
pyenv install -v 3.7.3
```

Once this is done, set up the environment and install the required libraries:

```
pyenv virtualenv 3.7.3 event_flow
pyenv activate event_flow

pip install --upgrade pip==20.0.2

cd event_flow/
pip install -r requirements.txt
```

### Download datasets

In this work, we use multiple datasets:
- 


In this project we use [MLflow](https://www.mlflow.org/docs/latest/index.html#) to keep track of the experiments. To visualize the models that are available, alongside other useful details and evaluation metrics, run the following from the home directory of the project:

```
mlflow ui
```

and access [http://127.0.0.1:5000](http://127.0.0.1:5000) from your browser of choice.

## Inference

To perform motion segmentation from event sequences from the dataset, run:

```
python eval_flow.py <model_name> --config configs/eval_MVSEC.yml

# for example:
python eval_flow.py LIFFireNet --config configs/eval_MVSEC.yml
```

Results from these evaluations are stored as MLflow artifacts. 

In `configs/`, you can find the configuration files associated to these scripts and vary the inference settings (e.g., number of input events, activate/deactivate visualization).

## Training

Run:

```
python train_flow.py --config configs/train_ANN.yml
```
 

**Note that we used a batch size of 8 in our experiments. Depending on your computational resources, you may need to lower this number.**

During and after the training, information about your run can be visualized through MLflow.

## Uninstalling pyenv

Once you finish using our code, you can uninstall `pyenv` from your system by:

1. Removing the `pyenv` configuration lines from your `~/.bashrc`.
2. Removing its root directory. This will delete all Python versions that were installed under the `$HOME/.pyenv/versions/` directory:

```
rm -rf $HOME/.pyenv/
```
