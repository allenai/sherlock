import argparse
import numpy as np
import torch
import json
import pprint
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomGrayscale, ColorJitter
import tempfile
import tqdm
import os
import collections
import clip
import sklearn.metrics
from scipy.stats import rankdata
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from frozendict import frozendict


class CLIPDatasetLeaderboard(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.args = args
        self.data = data
        self.id2data = {d['instance_id']: d for d in self.data}
        self.preprocess = self._transform_test(args.input_resolution)

    def url2filepath(self, url):
        if 'VG_' in url:
            return self.args.vg_dir + '/'.join(url.split('/')[-2:])
        else:
            # http://s3-us-west-2.amazonaws.com/ai2-rowanz/vcr1images/lsmdc_3023_DISTRICT_9/3023_DISTRICT_9_01.21.02.808-01.21.16.722@5.jpg
            if 'vcr1images' in self.args.vcr_dir:
                return self.args.vcr_dir + '/'.join(url.split('/')[-2:])
            else:
                return self.args.vcr_dir + '/'.join(url.split('/')[-3:])

    def hide_region(self, image, bboxes):
        image = image.convert('RGBA')
        if self.args.hide_true_bbox in [2,8]:
            overlay = Image.new('RGBA', image.size, '#00000000')
            draw = ImageDraw.Draw(overlay, 'RGBA')
        for bbox in bboxes:
            x = bbox['left']
            y = bbox['top']
            if self.args.hide_true_bbox in [2,8]: # highlight mode
                draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                               fill='#ff05cd3c', outline='#05ff37ff', width=3)
        if self.args.hide_true_bbox in [2,8]:
            image = Image.alpha_composite(image, overlay)

        return image

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


    def image_to_torch_tensor(self, image):
        width, height = image.size
        if width >= height:
            im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
            im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
        else:
            im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
            im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
        regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]
        image = torch.stack([self.preprocess(r) for r in regions], 0)
        return image


    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(self.url2filepath(c_data['image']['url']))
        image = self.hide_region(image, c_data['region'])

        if self.args.hide_true_bbox == 8:
            caption = clip.tokenize('inference: {}'.format(c_data['inference']), truncate=True).squeeze()
        else:
            caption = clip.tokenize(c_data['inference'], truncate=True).squeeze()

        cid = c_data['instance_id']

        image = self.image_to_torch_tensor(image)

        return {'image':image, 'caption':caption, 'id': cid}

    def __len__(self):
        return len(self.data)


class CLIPDatasetImageOnlyLeaderboard(CLIPDatasetLeaderboard):
    def __init__(self, images, args):
        self.args = args
        # images must contain [{'url': ..., 'bboxes': [ ... ]}]
        self.images = images
        self.preprocess = self._transform_test(args.input_resolution)

    def __getitem__(self, idx):
        c_data = self.images[idx]
        image = Image.open(self.url2filepath(c_data['url']))
        image = self.hide_region(image, c_data['bboxes'])
        image = self.image_to_torch_tensor(image)
        return {'image': image}

    def __len__(self):
        return len(self.images)


class CLIPDatasetCaptionOnlyLeaderboard(CLIPDatasetLeaderboard):
    def __init__(self, captions, args):
        self.args = args
        # images must contain [{'inference': ...}]
        self.captions = captions

    def __getitem__(self, idx):
        c_data = self.captions[idx]
        if self.args.hide_true_bbox == 8:
            caption = clip.tokenize('inference: {}'.format(c_data['caption']), truncate=True).squeeze()
        else:
            caption = clip.tokenize(c_data['caption'], truncate=True).squeeze()

        return {'caption': caption}

    def __len__(self):
        return len(self.captions)

    
class CLIPDatasetLocalizationLeaderboard(CLIPDatasetLeaderboard):
    def __init__(self, url2instances, args):
        self.args = args
        self.url2instances = url2instances
        self.ordered_urls = list(self.url2instances.keys())
        self.preprocess = self._transform_test(args.input_resolution)

    def __getitem__(self, idx):
        c_url = self.ordered_urls[idx]
        c_instances = self.url2instances[c_url]
        unique_inferences = list(set([inst['inference'] for inst in c_instances]))
        unique_regions = list(set([tuple([frozendict(x) for x in inst['region']]) for inst in c_instances]))

        inf2idx = {inf: idx for idx, inf in enumerate(unique_inferences)}
        reg2idx = {reg: idx for idx, reg in enumerate(unique_regions)}

        # assume we are going to do image x caption similarity
        # we want the lookup for each instance id
        # we can return a list of instance ids
        # a list of rows (image idxs)
        # and a list of cols (cap idxs)
        # then, we can zip lookup later.
        instance_ids = [inst['test_id'] for inst in c_instances]
        image_idxs = [reg2idx[tuple([frozendict(x) for x in inst['region']])] for inst in c_instances]
        cap_idxs = [inf2idx[inst['inference']] for inst in c_instances]
            
        if self.args.hide_true_bbox == 8:
            caption = clip.tokenize(['inference: {}'.format(cap) for cap in unique_inferences], truncate=True)
        else:
            caption = clip.tokenize(unique_inferences, truncate=True)
        
        image = Image.open(self.url2filepath(c_url))
        image = torch.stack([
            self.image_to_torch_tensor(self.hide_region(image, reg))
            for reg in unique_regions])

        return {'caption': caption, 'image': image, 'instance_ids': instance_ids, 'image_idxs': image_idxs, 'cap_idxs': cap_idxs}

    def __len__(self):
        return len(self.ordered_urls)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def clip_forward(model, image, text, only_features=True):
    if len(image.shape) == 5:
        im_feat1 = model.encode_image(image[:, 0, ...])
        im_feat2 = model.encode_image(image[:, 1, ...])
        image_features = (im_feat1 + im_feat2) / 2
    else:
        image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    if only_features:
        return image_features, text_features

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text, image_features, text_features


def clip_forward_image(model, image):
    if len(image.shape) == 5:
        im_feat1 = model.encode_image(image[:, 0, ...])
        im_feat2 = model.encode_image(image[:, 1, ...])
        image_features = (im_feat1 + im_feat2) / 2
    else:
        image_features = model.encode_image(image)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


def clip_forward_text(model, text):
    text_features = model.encode_text(text)
    # normalized features
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


class CLIPExtractor(torch.nn.Module):
    def __init__(self, clip_model, args):
        super(CLIPExtractor, self).__init__()
        self.clip_model = clip_model
        self.args = args

    def forward(self, image, text):
        return clip_forward(self.clip_model, image, text)

    def image_forward(self, image):
        return clip_forward_image(self.clip_model, image)

    def text_forward(self, text):
        return clip_forward_text(self.clip_model, text)
