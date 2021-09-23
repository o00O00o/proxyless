source /mnt/lustre/share/spring/r0.3.3

cd ..

# search in supervised mode
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=super_search --kill-on-bad-exit=1 \
python search.py --mode supervised \
--path /mnt/lustre/gaoyibo.vendor/proxyless/record/search_baseline/ \
--data_path /mnt/lustre/gaoyibo.vendor/Datasets/cifar-10/ \
--batch_size 256 \
--n_epochs 800

# train weight_preserving
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=super_preserve --kill-on-bad-exit=1 \
python train.py --weight_preserve --mode supervised \
--path /mnt/lustre/gaoyibo.vendor/proxyless/record/search_baseline/ \
--data_path /mnt/lustre/gaoyibo.vendor/Datasets/cifar-10/ \
--stablize_batch_size 256 \
--stablize_epochs 50 \
--batch_size 128 \
--n_epochs 600 &

# train from scratch
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=super_scratch --kill-on-bad-exit=1 \
python train.py --mode supervised \
--path /mnt/lustre/gaoyibo.vendor/proxyless/record/search_baseline/ \
--data_path /mnt/lustre/gaoyibo.vendor/Datasets/cifar-10/ \
--batch_size 128 \
--n_epochs 600 &
