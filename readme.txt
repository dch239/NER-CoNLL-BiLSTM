NOTES:

1) A folder data in the root folder must contains train, test, and dev files
 - data
    -train
    -test
    -dev

2) Extracted Glove Embeddings must be present in root directory
    - glove.6B.100d

3) main_script.py is the main codebase
    -There are two modes.

        -train: This mode trains both vanilla BiLSTM and Glove BiLSTM from scratch. 
            - Command: python main_script.py train
            - Generates: dev1.out, dev2.out, test1.out, test2.out. In addition it makes dev1_perl.out and dev2_perl.out which can be fed to the conll03eval script.

        -load: To use this mode, blstm1.pt and blstm2.pt must be in the root directory.
            - Command: python script.py load
            - Generates: dev1.out, dev2.out, test1.out, test2.out. In addition it makes dev1_perl.out and dev2_perl.out which can be fed to the conll03eval script.