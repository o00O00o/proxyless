source /mnt/lustre/share/spring/r0.3.3

cd ..

# search in SimCLR mode
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=SimCLR_search --kill-on-bad-exit=1 \
python search.py --mode SimCLR \
--path /mnt/lustre/gaoyibo.vendor/proxyless/record/search_SimCLR/ \
--data_path /mnt/lustre/gaoyibo.vendor/Datasets/cifar-10/ \
--batch_size 2048 \
--n_epochs 6400

# train weight_preserving
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=SimCLR_preserve --kill-on-bad-exit=1 \
python train.py --mode SimCLR --weight_preserve \
--path /mnt/lustre/gaoyibo.vendor/proxyless/record/search_SimCLR/ \
--data_path /mnt/lustre/gaoyibo.vendor/Datasets/cifar-10/ \
--stablize_batch_size 2048 \
--stablize_epochs 400 \
--batch_size 128 \
--n_epochs 600 &

# train from scratch
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=SimCLR_scratch --kill-on-bad-exit=1 \
python train.py --mode SimCLR \
--path /mnt/lustre/gaoyibo.vendor/proxyless/record/search_SimCLR/ \
--data_path /mnt/lustre/gaoyibo.vendor/Datasets/cifar-10/ \
--batch_size 128 \
--n_epochs 600 &
