#!/bin/bash

mkdir test_set_predictions

python predict_clip_leaderboard.py \
       ../test_comparison_public/test_comparison_instances.json \
       model=RN50x64~batch=64~warmup=1000~lr=1e-05~valloss=0.0000~randomclueinfhighlightbbox~widescreen_STEP=25200.pt \
       test_set_predictions/comparison.npy \
       --vcr_dir images/ \
       --vg_dir images/ \
       --clip_model RN50x64 \
       --hide_true_bbox 8 \
       --workers_dataloader 8 \
       --task comparison


python predict_clip_leaderboard.py \
       ../test_localization_public/test_localization_instances.json \
       model=RN50x64~batch=64~warmup=1000~lr=1e-05~valloss=0.0000~randomclueinfhighlightbbox~widescreen_STEP=25200.pt \
       test_set_predictions/localization.npy \
       --vcr_dir images/ \
       --vg_dir images/ \
       --clip_model RN50x64 \
       --hide_true_bbox 8 \
       --workers_dataloader 8 \
       --task localization


for split in {0..22};
do
    python predict_clip_leaderboard.py \
	   ../test_retrieval_public/test_retrieval_$split\_instances.json \
	   model=RN50x64~batch=64~warmup=1000~lr=1e-05~valloss=0.0000~randomclueinfhighlightbbox~widescreen_STEP=25200.pt \
	   test_set_predictions/retrieval_$split\.npy \
	   --vcr_dir images/ \
	   --vg_dir images/ \
	   --clip_model RN50x64 \
	   --hide_true_bbox 8 \
	   --workers_dataloader 8 \
	   --task retrieval
done
