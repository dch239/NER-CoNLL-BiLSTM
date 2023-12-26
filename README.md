# Project Title: Named Entity Recognition on CoNLL Dataset

This project is a Python-based implementation of a Bi-directional Long Short Term Memory (BLSTM) model for Named Entity Recognition (NER) tasks on the CoNLL dataset. The main script is `main_script.py`.

## Files in the Project

1. `main_script.py`: This is the main script of the project. It contains the implementation of the BLSTM model, the accuracy function for evaluating the model's performance, and the evaluate function for testing the model during training. The script is designed to perform NER on the CoNLL dataset.

2. `Walkthrough_Notebook.ipynb`: This is a Jupyter notebook that provides a step-by-step walkthrough of the project. It explains the code and the concepts used in the project in an interactive manner. It also provides a detailed explanation of the NER task and how it is performed on the CoNLL dataset.

3. `Walkthrough_PDF.pdf`: This is a PDF version of the Jupyter notebook. It can be used for offline reference.

4. `description.pdf`: This file contains the problem description that this project aims to solve. It provides a detailed explanation of the NER task and the expected solution.

5. `readme.txt`: This is a text file that provides a brief overview of the project. It also contains additional details about the project setup and execution.

## Setup Instructions

1. Clone the repository to your local machine.

2. Ensure that you have Python and the necessary libraries installed. The project requires `torch`, `numpy`, `sklearn`, and `tqdm`.

3. Download the CoNLL dataset and place it in the `data` folder in the root directory. The `data` folder should contain `train`, `test`, and `dev` files.

4. Download and extract the Glove Embeddings and place it in the root directory. The file should be named `glove.6B.100d`.

## How to Run the Project

The `main_script.py` file can be run in two modes:

1. `train`: This mode trains both vanilla BiLSTM and Glove BiLSTM from scratch. 
    - Command: `python main_script.py train`
    - Generates: `dev1.out`, `dev2.out`, `test1.out`, `test2.out`. In addition it makes `dev1_perl.out` and `dev2_perl.out` which can be fed to the conll03eval script.

2. `load`: To use this mode, `blstm1.pt` and `blstm2.pt` must be in the root directory.
    - Command: `python main_script.py load`
    - Generates: `dev1.out`, `dev2.out`, `test1.out`, `test2.out`. In addition it makes `dev1_perl.out` and `dev2_perl.out` which can be fed to the conll03eval script.

To understand the project in detail, go through the `Walkthrough_Notebook.ipynb` file. You can open this file in Jupyter Notebook.

If you prefer an offline reference, you can use the `Walkthrough_PDF.pdf` file.

For a detailed understanding of the NER task that this project aims to solve, refer to the `description.pdf` file.