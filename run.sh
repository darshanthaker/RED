#CUDA_VISIBLE_DEVICES=0 python experiments.py --dataset mnist --arch carlini_cnn --lambda1 0.00005 --lambda2 0.004 --test_lp 1 --solver prox --embedding scattering
#CUDA_VISIBLE_DEVICES=0 python experiments.py --dataset mnist --arch carlini_cnn --lambda1 0.00005 --lambda2 0.0003 --test_lp 1 --solver prox --embedding scattering
CUDA_VISIBLE_DEVICES=0 python experiments.py --dataset mnist --arch carlini_cnn --lambda1 0.999 --lambda2 0.999 --test_lp inf --solver active_refined
#CUDA_VISIBLE_DEVICES=1 python experiments.py --dataset synthetic --arch dense --lambda1 0.99 --lambda2 0.99 --test_lp 1 --solver active_refined --embedding warp
#CUDA_VISIBLE_DEVICES=0 python experiments.py --dataset cifar --arch carlini_cnn --lambda1 0.99 --lambda2 0.99 --test_lp inf --solver active_refined
