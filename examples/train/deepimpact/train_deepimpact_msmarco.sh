cudaNum=14

#### TRAIN ####
CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python train_deepimpact_msmarco.py