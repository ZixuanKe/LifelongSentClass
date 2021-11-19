



# Lifelong Sentiment Classification

This repository contains the code for the ECML-PKDD 2020 Paper [Continual Learning with Knowledge Transfer for Sentiment Classification](https://www.cs.uic.edu/~liub/publications/ECML-PKDD-2020.pdf) by Zixuan Ke, [Bing Liu](https://www.cs.uic.edu/~liub/), [Hao Wang](https://cshaowang.github.io/) and [Lei Shu](https://leishu02.github.io/). 

## News
[11/2021] **KAN** has been intergrated into [PyContinual](https://github.com/ZixuanKe/PyContinual). Check it out!
[11/2020] Interested in **Lifelong Learning/Continual Learning**? Check our latest NeurIPS 2020 paper [Continual Learning of a Mixed Sequence of Similar and Dissimilar Tasks](https://proceedings.neurips.cc/paper/2020/file/d7488039246a405baf6a7cbc3613a56f-Paper.pdf) and the [code](  
https://github.com/ZixuanKe/CAT) 

## Abstract
This paper studies continual learning (CL) for sentiment  classification (SC). In this setting, the CL system learns a sequence of  SC tasks incrementally in a neural network, where each task builds a classifier to classify the sentiment of reviews of a particular product category or domain. Two natural questions are: Can the system transfer the knowledge learned in the past from the previous tasks to the new task to help it learn a better model for the new task? And, can old models for previous tasks be improved in the process as well? This paper proposes a novel technique called KAN to achieve these objectives. KAN can markedly improve the SC accuracy of both the new task and the old tasks via forward and backward knowledge transfer. The effectiveness of KAN is demonstrated through extensive experiments

# Files
**/res**: all results saved in this folder  
**/dat**: processed toy dataset  
**/dataloader**: contained dataloader for the toy dataset   
**/reference**: additional code for baselines (specifically SRK and OWM)  
**/approaches**: code for training  
**/networks**: code for network architecture  

## Running
### Format
    run_train_[network]_[approach_specification]_[approach].sh
    [network]: lstm/gru
    [approch_specification]: optional, e.g. hat
    [approah]: ncl/one/mtl/
    [more options please refere to .sh files, run.py and config.py]
 
 ### Examples:
    run_train_bert_gru_kan_ncl.sh #Run KAN with GRU backbone

## Datasets
The asc (aspect-level sentiment classification), ae (aspect extraction) and dsc (document-level sentiment classification) datasets are given in **./dat**.

## Reference
If using this code, parts of it, or developments from it, please cite the reference bellow.

    @inproceedings{ke2020continual,
    author= {Zixuan Ke and Bing Liu and Hao Wang and Lei Shu},
    title= {Continual Learning with Knowledge Transfer for Sentiment Classification},
    booktitle = {ECML-PKDD},
    year = {2020}}

## Contact

Please drop an email to [Zixuan Ke](zke4@uic.edu) if you have any questions. 


## Acknowledgments
 [HAT](https://github.com/joansj/hat/tree/master/src)  
