source /mnt/lustre/share/spring/r0.3.3

cd ..

srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=2d_tr --kill-on-bad-exit=1 \
python search.py \
--path /mnt/lustre/gaoyibo.vendor/proxyless/record/search_baseline/ \
--data_path /mnt/lustre/gaoyibo.vendor/Datasets/cifar-10/ \
--batch_size 512 \
--mode SimCLR \
