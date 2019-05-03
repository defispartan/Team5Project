# Team5Project
CSE 842 Final Project Repository for Reproducing Experiments


**BrsDNC**
Original codebase: https://github.com/JoergFranke/ADNC

After installing dependencies found in requirements.txt, the following commands can be called:

  python scripts/inference_babi_task.py <model>
  
    Model Options: biadnc-all, biadnc-aug16-all
    
    Note: cse842 model was trained using the best configuration described in project report
    
  
  python scripts/start_training.py
  
    Configuration can be altered in scripts/config.yml
    
  
  
**BigBird (MT-DNN)** Original codebase: https://github.com/namisan/mt-dnn

Our results and model predictions are provided in the BigBird/Results directory, and the examples we used for error analysis are in the [BigBird/error_analysis](BigBird/error_analysis) directory.

After installing the model as documented in the MT-DNN repository, replace the following files with the identically named files in the [BigBird](BigBird) directory of our repository:
* data_utils/[glue_utils.py](BigBird/glue_utils.py) (data loading code modified for Winograd and COPA)
* data_utils/[label_map.py](BigBird/label_map.py) (modifed to provide some task-specific options for Winograd and COPA)
* [prepro.py](BigBird/prepro.py) (data preprocessing code modified for Winograd and COPA)
* [train.py](BigBird/train.py) (training code, modified slightly to get test results easier)

To reproduce our results on Winograd and COPA, perform the following steps from the root directory of the MT-DNN repository:
1. Preprocess the Winograd and COPA data. Running the following command will generate the preprocessed data files in the mt-dnn/data/mt-dnn directory:
```
python prepro.py
```
2. Copy the [bigbird_da_winograd.sb](BigBird/bigbird_da_winograd.sb) and [bigbird_da_copa.sb](BigBird/bigbird_da_copa.sb) files to the mt-dnn/scripts directory. In each file, replace the Python instance path on line 36 with the path to your own instance.

3. Submit both files as batch jobs to the HPCC. This can be done using the command:
```
sbatch bigbird_da_winograd.sb
sbatch bigbird_da_copa.sb
```

4. After no more than 8 hours of running, the jobs should complete. In the mt-dnn/scripts/checkpoints directory, you should see a directory for each training epoch for each benchmark. The reproduced results for each benchmark will be in the last epoch directory, as well as the learned weights. The evaluation metrics for COPA will be listed in a file called "copa_test_scores_13.json", while the metrics for Winograd will be listed in a file called "winograd_test_scores_9.json".

5. To evaluate Winograd separately for each test set (by default it is evaluated on the union of both test sets) and perform error analysis, move the "winograd_test_scores_9.json" file to the [BigBird/error_analysis](BigBird/error_analysis) directory of our repository and rename it "winograd_pred.json". Then run the following command to print the accuracy on each test set, as well as generate files listing the correctly and incorrectly classified examples in the test sets:
```
python error_analysis/error_analysis_winograd.py
```

6. To perform error analysis on COPA, move the "copa_test_scores_13.tsv" file to the [BigBird/error_analysis](BigBird/error_analysis) directory of our repository and rename it "copa_pred.tsv". Then run the following command to generate files listing the correctly and incorrectly classified examples in the test set:
```
python error_analysis/error_analysis_copa.py
```
**GPT** 

Original codebase: https://github.com/openai/finetune-transformer-lm

Check for the correct instance of python in following files.
GPT/COPA/gpt_copa.sb
GPT/Winograd/gpt_winograd.sb
These files will be used to build model using HPCC.

**GPT Experiment for COPA**
To experiment with COPA, use the training and testing data in [GPT/COPA/data](GPT/COPA/data).
Replace the datasets.py with [GPT/COPA/datasets.py](GPT/COPA/datasets.py) and use the correct directory path for training and test file.
Update the test file path in analysis.py file
Update the train.py with the hyperparameter such as number of epochs, batch size, learning rate.
```
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--n_batch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=6.25e-5)
```
Submit both files as batch jobs to the HPCC. This can be done using the command:
```
sbatch gpt_copa.sb
```
**GPT Experiment for Winograd**
To experiment with Winograd we need use the training and testing data in [GPT/Winograd/data](GPT/Winograd/data). Replace the datasets.py with [GPT/Winograd/datasets.py](GPT/Winograd/datasets.py) and use the correct directory path for training and test file.
Update the test file path in analysis.py file.
Update the train.py with the hyperparameter such as number of epochs, batch size, learning rate.
```
    parser.add_argument('--n_iter', type=int, default=5)
    parser.add_argument('--n_batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
```
Submit both files as batch jobs to the HPCC. This can be done using the command.

```
sbatch gpt_winograd.sb
```


A job_id will be generated and a log file in the name of "slurm-job_id.out" will be generated in the same working directory.

