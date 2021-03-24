#CUDA_VISIBLE_DEVICES=2 python experiments.py --dataset yale --arch dense --lambda1 0.2 --lambda2 0.1 --test_lp inf
CUDA_VISIBLE_DEVICES=2 python experiments.py --dataset mnist --arch carlini_cnn --lambda1 5 --lambda2 15 --test_lp inf
#CUDA_VISIBLE_DEVICES=1 python sbsc.py --dataset yale --arch dense
