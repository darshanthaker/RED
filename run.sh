#CUDA_VISIBLE_DEVICES=2 python experiments.py --dataset yale --arch dense --lambda1 0.2 --lambda2 0.1 --test_lp 2 --regularizer 4
CUDA_VISIBLE_DEVICES=2 python experiments.py --dataset mnist --arch carlini_cnn --lambda1 5 --lambda2 10 --test_lp inf --embedding scattering --del_threshold 0.2
#CUDA_VISIBLE_DEVICES=1 python sbsc.py --dataset yale --arch dense
