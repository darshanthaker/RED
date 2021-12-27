#CUDA_VISIBLE_DEVICES=2 python experiments.py --dataset yale --arch dense --lambda1 0.2 --lambda2 0.1 --test_lp 2 --regularizer 4
#CUDA_VISIBLE_DEVICES=0 python experiments.py --dataset mnist --arch carlini_cnn --lambda1 5 --lambda2 15 --test_lp inf --del_threshold 0.2 --regularizer 4 --solver irls
CUDA_VISIBLE_DEVICES=0 python experiments.py --dataset mnist --arch carlini_cnn --lambda1 0.99 --lambda2 0.99 --test_lp 1 --solver active_refined
#CUDA_VISIBLE_DEVICES=1 python experiments.py --dataset synthetic --arch dense --lambda1 0.99 --lambda2 0.99 --test_lp 1 --solver active_refined --embedding warp
#CUDA_VISIBLE_DEVICES=0 python experiments.py --dataset cifar --arch carlini_cnn --lambda1 0.99 --lambda2 0.99 --test_lp inf --solver active_refined
#CUDA_VISIBLE_DEVICES=0 python experiments.py --dataset mnist --arch carlini_cnn --lambda1 0.99 --lambda2 0.99 --test_lp inf --solver active_refined --embedding scattering --use_cheat_grad
