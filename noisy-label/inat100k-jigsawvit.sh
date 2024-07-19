CUDA_VISIBLE_DEVICES=0,1 python3 jigsaw-vit/main.py \
  --arch 'vit_small_patch16' \
   --input-size 224 \
   --batch-size 128 \
   --max-lr 1e-3 \
   --min-lr 1e-6 \
   --total-iter 400000 \
   --warmup-iter 20000 \
   --niter-eval 10000 \
   --weight-decay 0.05 \
   --smoothing 0.1 \
   --mixup 0.8 \
   --cutmix 1.0 \
   --world-size 1 \
   --workers 16 \
   --rank 0 \
   --data-path /home/aoneill/data/iNat100k \
   --data-set 'inat100k' \
   --eta 2.0 \
   --mask-ratio 0.1 \
   --out-dir /home/trode/jigsaw-vit-noisy/out
