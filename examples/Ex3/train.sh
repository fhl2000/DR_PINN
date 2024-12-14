# default setting
# CUDA_VISIBLE_DEVICES=6 python coupling.py --seed 0
# CUDA_VISIBLE_DEVICES=6 python decoupling_C.py --seed 0
# CUDA_VISIBLE_DEVICES=6 python decoupling_A.py --seed 0
# CUDA_VISIBLE_DEVICES=6 python decoupling_B.py --seed 0


# ablation of architecture
# CUDA_VISIBLE_DEVICES=0 python coupling.py     --network Modified --seed 0
# CUDA_VISIBLE_DEVICES=0 python decoupling_C.py   --network Modified --seed 0
# CUDA_VISIBLE_DEVICES=0 python decoupling_A.py          --network Modified --seed 0
# CUDA_VISIBLE_DEVICES=0 python decoupling_B.py       --network Modified --seed 0
# CUDA_VISIBLE_DEVICES=0 python coupling.py     --network Shallow --seed 0
# CUDA_VISIBLE_DEVICES=0 python decoupling_C.py   --network Shallow --seed 0
# CUDA_VISIBLE_DEVICES=0 python decoupling_A.py          --network Shallow --seed 0
# CUDA_VISIBLE_DEVICES=0 python decoupling_B.py       --network Shallow --seed 0
# CUDA_VISIBLE_DEVICES=0 python coupling.py     --network Deep --seed 0
# CUDA_VISIBLE_DEVICES=0 python decoupling_C.py   --network Deep --seed 0
# CUDA_VISIBLE_DEVICES=0 python decoupling_A.py          --network Deep --seed 0
# CUDA_VISIBLE_DEVICES=0 python decoupling_B.py       --network Deep --seed 0

# ablation of optimization
# CUDA_VISIBLE_DEVICES=3 python coupling.py     --optimizer 0 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_C.py   --optimizer 0 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_A.py          --optimizer 0 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_B.py       --optimizer 0 --seed 0
# CUDA_VISIBLE_DEVICES=3 python coupling.py     --optimizer 1 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_C.py   --optimizer 1 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_A.py          --optimizer 1 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_B.py       --optimizer 1 --seed 0
# CUDA_VISIBLE_DEVICES=3 python coupling.py     --optimizer 2 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_C.py   --optimizer 2 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_A.py          --optimizer 2 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_B.py       --optimizer 2 --seed 0
# CUDA_VISIBLE_DEVICES=3 python coupling.py     --optimizer 3 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_C.py   --optimizer 3 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_A.py          --optimizer 3 --seed 0
# CUDA_VISIBLE_DEVICES=3 python decoupling_B.py       --optimizer 3 --seed 0


