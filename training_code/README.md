# CLIP-based model training code for sherlock.

While there are several pretrained models available, for those
interested in replicating our training process, we additionally
provide our code. To train, please follow these instructions:

1. Download the VG and VCR images (see this repo's main README for guidance) and put them in `images/`
2. Download the sherlock training/validation annotations (see this repo's main README for guidance)
3. Run `train_retrieval_clip.py` with any options you like. The command to replicate our most performant multitask model:

```
python train_retrieval_clip.py \
       sherlock_train_v1.1.json \
       sherlock_val_with_split_idxs_v1.1.json \
       --workers_dataloader 16 \
       --clip_model RN50x64 \
       --lr .00001 \
       --batch_size 64 \
       --warmup 1000 \
       --n_epochs 5 \
       --hide_true_bbox 8 \ # see training args for different modes, this is multitask + highlighting mode
       --widescreen_processing 1 \
       --output_dir clip_multitask
```

More options can be seen with `python train_retrieval_clip.py --help`
--- feel free to reach out if you have questions about particular
settings.

Note that, to train with the specified batch size, you may need up to
8 48GB GPUs --- if you have less compute available, consider training
with a more efficient model like `ViT/B-16`.