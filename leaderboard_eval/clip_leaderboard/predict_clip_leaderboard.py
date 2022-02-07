'''
Prediction script for leaderboard
'''
import argparse
import numpy as np
import torch
import json
import pprint
from PIL import Image, ImageDraw
import tempfile
import tqdm
import os
import collections
import clip
import sklearn.metrics
from scipy.stats import rankdata
import sys
import clip_inference_iterator
import pprint
from frozendict import frozendict

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('instances')

    parser.add_argument('load_model_from')

    parser.add_argument('output_npy')

    parser.add_argument('--task',
                        default='retrieval',
                        choices=['retrieval',
                                 'localization',
                                 'comparison'])

    parser.add_argument('--clip_model',
                        default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'RN50x16', 'RN50x64'])

    parser.add_argument('--batch_size',
                        default=256,
                        type=int,
                        help='batch size. due to numerical precision annoyance, keep at 256 for exact replication.')

    parser.add_argument(
        '--vcr_dir',
        default='/net/nfs2.mosaic/jackh/vcr_images/vcr1images/',
        help='directory with all of the VCR image data, contains, e.g., movieclips_Lethal_Weapon')

    parser.add_argument(
        '--vg_dir',
        default='/net/nfs2.mosaic/jackh/extract_butd_image_features_sherlock/',
        help='directory with visual genome data, contains VG_100K and VG_100K_2')

    parser.add_argument('--hide_true_bbox',
                        type=int,
                        default=2,
                        choices=[0,2,8],
                        help='0=nohighlight, 2=singletask, 8=multitask')

    parser.add_argument('--workers_dataloader',
                        type=int,
                        default=8)

    args = parser.parse_args()

    if args.vcr_dir[-1] != '/':
        args.vcr_dir += '/'
    if args.vg_dir[-1] != '/':
        args.vg_dir += '/'

    if os.path.exists(args.output_npy):
        print('{} already exists! continuing'.format(args.output_npy))
        quit()

    return args


def comparison_main(model, args):
    with open(args.instances) as f:
        instances = json.load(f)
    print('{} comparison instances'.format(len(instances)))
    for inst in instances:
        inst['instance_id'] = inst['test_id']

    dset = torch.utils.data.DataLoader(
        clip_inference_iterator.CLIPDatasetLeaderboard(instances, args),
        batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

    all_im_embs, all_txt_embs = [], []
    with torch.no_grad():
        bar = tqdm.tqdm(dset, total=len(dset))
        for batch in bar:
            images, captions = batch['image'].to(args.device), batch['caption'].to(args.device)
            image_features, text_features = model(images, captions)
            all_im_embs.append(image_features)
            all_txt_embs.append(text_features)

    all_im_embs = torch.cat(all_im_embs).cpu()
    all_txt_embs = torch.cat(all_txt_embs).cpu()
    # this is *not* the most efficient way to do this but,
    # nonetheless, for reproducability, this is how this was computed
    # using the other setup.
    im2text_sim = -sklearn.metrics.pairwise_distances(all_im_embs,
                                                      all_txt_embs,
                                                      metric='cosine',
                                                      n_jobs=args.workers_dataloader) + 1
    sims = np.diag(im2text_sim).tolist()
    result = {}
    for inst, sim in zip(instances, sims):
        result[inst['test_id']] = sim

    sorted_scores = np.array(
        [result[k] for k in sorted(result.keys())]).astype(np.float32)
    np.save(args.output_npy, sorted_scores)

    print('writing {} to {}'.format(len(result), args.output_npy))

    
def retrieval_main(model, args):

    with open(args.instances) as f:
        instances = json.load(f)
    print('{} retrieval instances'.format(len(instances)))

    all_images, all_inferences = set(), set()
    for d in tqdm.tqdm(instances):
        cur_image = frozendict({'url': d['image']['url'], 'bboxes': tuple([frozendict(r) for r in d['region']])})
        cur_inference = frozendict({'caption': d['inference']})
        all_images.add(cur_image)
        all_inferences.add(cur_inference)

    all_images = list(all_images)
    im2idx = {k: idx for idx, k in enumerate(all_images)}
    all_inferences = list(all_inferences)
    inf2idx = {k: idx for idx, k in enumerate(all_inferences)}

    image_dataloader = torch.utils.data.DataLoader(
        clip_inference_iterator.CLIPDatasetImageOnlyLeaderboard(all_images, args),
        batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

    inference_dataloader = torch.utils.data.DataLoader(
        clip_inference_iterator.CLIPDatasetCaptionOnlyLeaderboard(all_inferences, args),
        batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

    all_im_embs, all_txt_embs = [], []
    with torch.no_grad():
        bar = tqdm.tqdm(image_dataloader, total=len(image_dataloader))
        for batch in bar:
            images = batch['image'].to(args.device)
            im_embs = model.image_forward(images)
            all_im_embs.append(im_embs.cpu())
        bar = tqdm.tqdm(inference_dataloader, total=len(inference_dataloader))
        for batch in bar:
            captions = batch['caption'].to(args.device)
            txt_embs = model.text_forward(captions)
            all_txt_embs.append(txt_embs)

    all_im_embs = torch.cat(all_im_embs).cpu()
    all_txt_embs = torch.cat(all_txt_embs).cpu()
    im2text_sim = -sklearn.metrics.pairwise_distances(all_im_embs,
                                                      all_txt_embs,
                                                      metric='cosine',
                                                      n_jobs=args.workers_dataloader) + 1
    result = {}
    for d in tqdm.tqdm(instances):
        cur_image = frozendict({'url': d['image']['url'], 'bboxes': tuple([frozendict(r) for r in d['region']])})
        cur_inference = frozendict({'caption': d['inference']})
        im_idx, inf_idx = im2idx[cur_image], inf2idx[cur_inference]
        result[d['test_id']] = im2text_sim[im_idx, inf_idx]

    print('writing {} to {}'.format(len(result), args.output_npy))

    sorted_scores = np.array(
        [result[k] for k in sorted(result.keys())]).astype(np.float32)
    np.save(args.output_npy, sorted_scores)


def localization_main(model, args):
    with open(args.instances) as f:
        instances = json.load(f)
    print('{} localization instances'.format(len(instances)))

    # for the paper, we floatified the model for localization. it
    # doesn't matter much, but for reproducability...  while all the
    # result differences are very small, somewhat surprisingly,
    # float16 seems to be less deterministic. Possibly should consider
    # floatifying everything; that's how training was done after all.
    model.float()

    url2instances = collections.defaultdict(list)
    all_images, all_inferences = set(), set()
    for d in tqdm.tqdm(instances):
        url2instances[d['image']['url']].append(d)

    localization_dataloader = torch.utils.data.DataLoader(
        clip_inference_iterator.CLIPDatasetLocalizationLeaderboard(url2instances, args),
        batch_size=1, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

    result = {}
    with torch.no_grad():
        bar = tqdm.tqdm(localization_dataloader, total=len(localization_dataloader))
        for batch in bar:
            im_embs = model.image_forward(batch['image'].to(args.device).squeeze(0))
            txt_embs = model.text_forward(batch['caption'].to(args.device).squeeze(0))
            im2txt = im_embs @ torch.transpose(txt_embs, 0, 1)
            im2txt = im2txt.cpu().numpy()
            for inst_id, im_idx, txt_idx in zip(batch['instance_ids'],
                                                batch['image_idxs'],
                                                batch['cap_idxs']):
                result[inst_id[0]] = im2txt[im_idx.item(), txt_idx.item()]

    print('writing {} to {}'.format(len(result), args.output_npy))
    sorted_scores = np.array(
        [result[k] for k in sorted(result.keys())]).astype(np.float32)
    np.save(args.output_npy, sorted_scores)


def main():
    args = parse_args()
    np.random.seed(1)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(args.clip_model, jit=False)

    if args.load_model_from != 'None':
        print('Getting model weights from {}'.format(args.load_model_from))
        state = torch.load(args.load_model_from, map_location=args.device)
        state['model_state_dict'] = {k.replace('module.clip_model.', '') : v for k, v in state['model_state_dict'].items()}
        state['model_state_dict'] = {k.replace('clip_model.', '') : v for k, v in state['model_state_dict'].items()}
        model.load_state_dict(state['model_state_dict'])

    try:
        args.input_resolution = model.visual.input_resolution
    except:
        args.input_resolution = model.input_resolution

    model = clip_inference_iterator.CLIPExtractor(model, args)
    model.to(args.device)
    model.eval()

    if args.task == 'retrieval':
        retrieval_main(model, args)
    elif args.task == 'localization':
        localization_main(model, args)
    elif args.task == 'comparison':
        comparison_main(model, args)


if __name__ == '__main__':
    main()
