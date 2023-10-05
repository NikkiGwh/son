# SON-Simulation

A mobile Network simulation

- [Documentation](https://www.youtube.com/watch?v=VQyViaC_QOg&ab_channel=OzzyManReviews)

![topologyimage](topologyimage.png)

## Requirements

- Python 3.9+

## How to install

### Run this command in a command prompt when using normal python environment:

```
pip install -r pip-requirements.txt
```

### Run this command when using conda to create conda-environment with required packages

```
conda env create -f son-simulation-conda-env.yml
conda activate son-simulation-conda-env
```

## running normal interacitve UI-mode

just run

```
python pygame_editor.py
```

## example for running predefined experiments via script arguments

python network_simulation_script.py [network-name] [configuration-name]

```
python network_simulation_script.py hetNet2 hetNet2_predefined_70_greedy
```

predefined configurations (i.e. hetNet2_predefined_70_greedy.json) are located in the ./predefined_configs directory

You can only choose network names which already have a directory (i.e. hetNet2)
