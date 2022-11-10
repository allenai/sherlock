'''
Training model
'''
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
import torchvision.transforms.functional as F


class SquarePad:
    # https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/9
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, data, args, training=False):
        self.args = args
        self.data = data
        self.id2data = {d['instance_id']: d for d in self.data}
        self.training = training
        if self.args.widescreen_processing in [0, 1]:
            self.preprocess = self._transform_train(args.input_resolution) if self.training else self._transform_test(args.input_resolution)
        else:
            self.preprocess = self._transform_train_pad(args.input_resolution) if self.training else self._transform_test_pad(args.input_resolution)

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
        if self.args.hide_true_bbox == 1: # hide mode
            draw = ImageDraw.Draw(image, 'RGBA')
        if self.args.hide_true_bbox in [2,5,7,8,9]: #highlight mode
            overlay = Image.new('RGBA', image.size, '#00000000')
            draw = ImageDraw.Draw(overlay, 'RGBA')
        if self.args.hide_true_bbox == 3 or self.args.hide_true_bbox == 6: #blackout mode or position only mode
            overlay = Image.new('RGBA', image.size, '#7B7575ff')
            draw = ImageDraw.Draw(overlay, 'RGBA')
        for bbox in bboxes:
            x = bbox['left']
            y = bbox['top']
            if self.args.hide_true_bbox == 1: # hide mode
                draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])], fill='#7B7575')
            elif self.args.hide_true_bbox in [2,5,7,8,9]: # highlight mode
                draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                               fill='#ff05cd3c', outline='#05ff37ff', width=3)
            elif self.args.hide_true_bbox == 3: # blackout mode
                draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                               fill='#00000000')
            elif self.args.hide_true_bbox == 6: # position only mode
                draw.rectangle([(x, y), (x+bbox['width'], y+bbox['height'])],
                               fill='#ff05cdff')

        if self.args.hide_true_bbox in [2, 3, 5, 6, 7, 8, 9]:
            image = Image.alpha_composite(image, overlay)

        return image

    def _transform_train(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomCrop(n_px),
            RandomHorizontalFlip(),
            #RandomGrayscale(), # these were used in the model in the paper, but, something seems bugged when pytorch updated.
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_train_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            RandomHorizontalFlip(),
            #RandomGrayscale(), # these were used in the model in the paper, but, something seems bugged when pytorch updated.
            ColorJitter(brightness=.5, hue=.3),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _transform_test_pad(self, n_px):
        return Compose([
            SquarePad(),
            Resize(n_px, interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def image_to_torch_tensor(self, image):
        if self.args.widescreen_processing == 1:
            width, height = image.size
            if width >= height:
                im1 = {'height': height, 'width': height, 'left': 0, 'top': 0}
                im2 = {'height': height, 'width': height, 'left': width-height, 'top': 0}
            else:
                im1 = {'height': width, 'width': width, 'left': 0, 'top': 0}
                im2 = {'height': width, 'width': width, 'left': 0, 'top': height-width}
            regions = [image.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])) for bbox in [im1, im2]]
            image = torch.stack([self.preprocess(r) for r in regions], 0)
        else:
            image = self.preprocess(image)
        return image

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(self.url2filepath(c_data['inputs']['image']['url']))

        if self.args.hide_true_bbox > 0:
            image = self.hide_region(image, c_data['inputs']['bboxes'])

        clue = clip.tokenize(c_data['inputs']['clue'], truncate=True).squeeze()

        if self.args.hide_true_bbox == 5:
            caption = clip.tokenize('clue: {} infer: {}'.format(c_data['inputs']['clue'],
                                                                c_data['targets']['inference']),
                                    truncate=True).squeeze()
        elif self.args.hide_true_bbox == 7:
            caption = clip.tokenize('{}'.format(c_data['inputs']['clue']),
                                    truncate=True).squeeze()
        elif self.args.hide_true_bbox == 8:
            if self.training and np.random.random() < .5:
                caption = clip.tokenize('clue: {}'.format(c_data['inputs']['clue']), truncate=True).squeeze()
            else:
                caption = clip.tokenize('inference: {}'.format(c_data['targets']['inference']), truncate=True).squeeze()

        elif self.args.hide_true_bbox == 9:
            caption = clip.tokenize('clue: {}'.format(c_data['inputs']['clue']), truncate=True).squeeze()

        else:
            caption = clip.tokenize(c_data['targets']['inference'], truncate=True).squeeze()

        cid = c_data['instance_id']
        image = self.image_to_torch_tensor(image)
        return {'image':image, 'caption':caption, 'clue': clue, 'id': cid}

    def get(self, cid):
        return self.id2data[cid]

    def __len__(self):
        return len(self.data)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train')
    parser.add_argument('val')

    parser.add_argument('--clip_model',
                        default='ViT-B/16',
                        choices=['ViT-B/32', 'RN50', 'RN101', 'RN50x4', 'ViT-B/16', 'RN50x16', 'RN50x64', 'ViT-L/14@336px'])

    parser.add_argument('--batch_size',
                        type=int,
                        default=256)

    parser.add_argument('--warmup',
                        type=int,
                        default=1000)

    parser.add_argument('--init_from',
                        type=str,
                        default=None,
                        help='if set, model weights will be init from this set.')

    parser.add_argument(
        '--vcr_dir',
        default='images/',
        help='directory with all of the VCR image data, contains, e.g., movieclips_Lethal_Weapon')

    parser.add_argument(
        '--vg_dir',
        default='images/',
        help='directory with visual genome data, contains VG_100K and VG_100K_2')

    parser.add_argument('--lr',
                        type=float,
                        default=.00001)

    parser.add_argument('--n_epochs',
                        type=int,
                        default=3)

    parser.add_argument('--output_dir',
                        type=str,
                        default='clip_model_outputs')

    parser.add_argument('--debug',
                        type=int,
                        default=0)

    parser.add_argument('--hide_true_bbox',
                        type=int,
                        default=0,
                        choices=[0,1,2,3,4,5,6,7,8,9],
                        help=('0=plain, 1=hide true, 2=highlight true, 3=blackout all but true, '
                              '4=clue as image, 5=highlight + clue-inf together in text, 6=positiononly, '
                              '7=highlighting and only clue in text, '
                              '8=clues and inferences selected randomly (multitask), '
                              '9=clue with prefix, to complement a model trained with 8'))

    parser.add_argument('--workers_dataloader',
                        type=int,
                        default=8)

    parser.add_argument('--widescreen_processing',
                        type=int,
                        help='if 1, then we will run CLIP twice over each image twice to get a bigger field of view, if 2 we will squarepad (less computation), if 0 we center crop (less computation)',
                        default=1,
                        choices=[0,1,2])

    parser.add_argument('--save_every',
                        type=int,
                        help='if >1, a checkpoint will be saved every this many gradient updates.',
                        default=0)

    parser.add_argument('--early_stop',
                        type=int,
                        help='if > 0, if the loss doesnt improve in this many epochs, quit.',
                        default=5)

    parser.add_argument('--negative_sample_cache',
                        type=int,
                        help='if > 0, cache this many previous batches for negative sample loss computation. this is an experimental feature not used in the paper.',
                        default=0)

    parser.add_argument('--val_stat',
                        type=str,
                        help='which stat should we use for early stopping?',
                        choices=['loss', 'meanrank'])

    args = parser.parse_args()

    if args.negative_sample_cache > 0:
        print('Training with a negative sample cache. N batches will be computed in inference mode, then 1 gradient update')
        print('Recommendation: update your number of epochs to be N+1 * current epochs. E.g., 3 epochs before --> 12 epochs with neg sample cache 3')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.output_model_path = '{}/model={}~batch={}~warmup={}~lr={}'.format(
        args.output_dir, args.clip_model.replace('/', ''), args.batch_size, args.warmup, args.lr)

    args.output_model_path += '~valloss={:.4f}'

    if args.hide_true_bbox == 1:
        args.output_model_path += '~hiddenbbox'
    if args.hide_true_bbox == 2:
        args.output_model_path += '~highlightbbox'
    if args.hide_true_bbox == 3:
        args.output_model_path += '~blackoutbbox'
    if args.hide_true_bbox == 4:
        args.output_model_path += '~clueasimage'
    if args.hide_true_bbox == 5:
        args.output_model_path += '~clueandinfhighlightbbox'
    if args.hide_true_bbox == 6:
        args.output_model_path += '~positiononly'
    if args.hide_true_bbox == 7:
        args.output_model_path += '~highlight_clueinstead'
    if args.hide_true_bbox == 8:
        args.output_model_path += '~randomclueinfhighlightbbox'
    if args.hide_true_bbox == 8:
        args.output_model_path += '~cluewithprefix'

    if args.widescreen_processing == 0:
        args.output_model_path += '~cropping'
    elif args.widescreen_processing == 1:
        args.output_model_path += '~widescreen'
    elif args.widescreen_processing == 2:
        args.output_model_path += '~padsquare'

    if args.negative_sample_cache > 0:
        args.output_model_path += '~negativesamplecache={}'.format(args.negative_sample_cache)

    args.output_predictions_path = args.output_model_path + '~predictions.json'

    if args.vcr_dir[-1] != '/':
        args.vcr_dir += '/'
    if args.vg_dir[-1] != '/':
        args.vg_dir += '/'
    return args


def clip_forward(model, image, text):
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

    return image_features, text_features


def clip_forward_text_only(model, clue, inference):
    clue_features = model.encode_text(clue)
    inference_features = model.encode_text(inference)

    # normalized features
    clue_features = clue_features / clue_features.norm(dim=-1, keepdim=True)
    inference_features = inference_features / inference_features.norm(dim=-1, keepdim=True)

    return clue_features, inference_features


class CLIPExtractor(torch.nn.Module):
    def __init__(self, clip_model, args):
        super(CLIPExtractor, self).__init__()
        self.clip_model = clip_model
        self.args = args

    def forward(self, image, text):
        return clip_forward(self.clip_model, image, text)


class CLIPTextOnlyExtractor(torch.nn.Module):
    def __init__(self, clip_model, args):
        super(CLIPTextOnlyExtractor, self).__init__()
        self.clip_model = clip_model
        self.args = args

    def forward(self, clue, inference):
        return clip_forward_text_only(self.clip_model, clue, inference)


def main():
    args = parse_args()
    np.random.seed(1)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(args.clip_model, jit=False)

    try:
        args.input_resolution = model.visual.input_resolution
    except:
        args.input_resolution = model.input_resolution

    if args.hide_true_bbox != 4:
        model = CLIPExtractor(model, args)
    else:
        model = CLIPTextOnlyExtractor(model, args)

    if args.init_from:
        print('Getting model weights from {}'.format(args.init_from))
        state = torch.load(args.init_from)
        state['model_state_dict'] = {k.replace('module.', '') : v for k, v in state['model_state_dict'].items()}
        model.load_state_dict(state['model_state_dict'])

    model.to(args.device)

    all_images = set()
    imagepair2count = collections.Counter()
    with open(args.train) as f:
        train = json.load(f)
        train = torch.utils.data.DataLoader(
            CLIPDataset(train, args, training=True),
            batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=True, worker_init_fn=worker_init_fn)

    with open(args.val) as f:
        val = json.load(f)
        val = torch.utils.data.DataLoader(
            CLIPDataset(val, args, training=False),
            batch_size=args.batch_size, num_workers=args.workers_dataloader, shuffle=False, worker_init_fn=worker_init_fn)

    use_multi = False
    if torch.cuda.device_count() > 1:
        use_multi = True
        print('Lets use', torch.cuda.device_count(), 'GPUs!')
        model = torch.nn.DataParallel(model)

    model.float()

    print('train/val {}/{}'.format(len(train), len(val)))

    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()

    optim = torch.optim.AdamW([p for p in model.parameters()], lr=args.lr)
    best_val_loss = np.inf
    not_improved_epoch = 0
    tmpfile = tempfile.NamedTemporaryFile()
    print('using tempfile {}'.format(tmpfile.name))

    def lr_lambda(current_step):
        if current_step < args.warmup:
            mul = float(current_step) / float(max(1.0, args.warmup))
        else:
            mul = 1.0
        return mul

    schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    # learning rate reducer with no patience.
    reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=0, verbose=1)
    optim_steps_so_far = 0
    gradient_steps_so_far = 0

    for epoch in range(args.n_epochs):
        print('Epoch {}'.format(epoch))
        for mode in ['train', 'val']:
            if mode == 'train':
                model.train()
                bar = tqdm.tqdm(enumerate(train), total=len(train))
            else:
                model.eval()
                bar = tqdm.tqdm(enumerate(val), total=len(val))

            negative_sample_cache = []
            all_val_im_embs, all_val_txt_embs, all_val_ids = [], [], []
            n, running_sum_loss, running_sum_within_batch_loss = 0, 0, 0

            ground_truth = torch.arange(args.batch_size, dtype=torch.long, device=args.device)
            for i, batch in bar:
                with torch.set_grad_enabled(mode=='train' and len(negative_sample_cache) == args.negative_sample_cache):
                    optim.zero_grad()

                    if args.hide_true_bbox != 4:
                        images, captions = batch['image'].to(args.device), batch['caption'].to(args.device)
                        image_features, text_features = model(images, captions)
                    else:
                        clues, inferences = batch['clue'].to(args.device), batch['caption'].to(args.device)
                        image_features, text_features = model(clues, inferences)
                    if use_multi:
                        logit_scale = model.module.clip_model.logit_scale.exp()
                    else:
                        logit_scale = model.clip_model.logit_scale.exp()

                    c_batch_size = image_features.shape[0]
                    if args.negative_sample_cache == 0 or mode == 'val':
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        logits_per_text = logits_per_image.t()
                        total_loss = (loss_img(logits_per_image, ground_truth[:c_batch_size]) +
                                      loss_txt(logits_per_text, ground_truth[:c_batch_size]))/2

                    if args.negative_sample_cache > 0:
                        if len(negative_sample_cache) == args.negative_sample_cache:
                            big_image_feats = torch.vstack([image_features] + [x[0] for x in negative_sample_cache])
                            big_text_feats = torch.vstack([text_features] +[x[1] for x in negative_sample_cache])

                            logits_per_image = logit_scale * image_features @ big_text_feats.t()
                            logits_per_text = logit_scale * text_features @ big_image_feats.t()

                            total_loss = (loss_img(logits_per_image, ground_truth[:c_batch_size]) +
                                          loss_txt(logits_per_text, ground_truth[:c_batch_size]))/2

                            within_batch_loss = (loss_img(logits_per_image[:, :c_batch_size], ground_truth[:c_batch_size]) +
                                                 loss_txt(logits_per_text[:, :c_batch_size], ground_truth[:c_batch_size]))/2
                            negative_sample_cache = []

                        else:
                            # update cache
                            negative_sample_cache.append((image_features.detach(), text_features.detach()))
                            continue

                    if mode == 'train':
                        total_loss.backward()
                        optim.step()

                        if args.save_every > 0 and gradient_steps_so_far % args.save_every == 0:
                            print('saving at step {} to {}'.format(gradient_steps_so_far, args.output_model_path.format(0.0) + '_STEP={}.pt'.format(gradient_steps_so_far)))
                            torch.save(
                                {'model_state_dict': model.state_dict()},
                                args.output_model_path.format(0.0) + '_STEP={}.pt'.format(gradient_steps_so_far))

                        gradient_steps_so_far += 1
                        if optim_steps_so_far < args.warmup:
                            optim_steps_so_far += 1
                            schedule.step()
                    elif mode == 'val':
                        all_val_im_embs.append(image_features)
                        all_val_txt_embs.append(text_features)
                        all_val_ids.extend(list(batch['id']))

                    n += 1
                    if args.negative_sample_cache > 0:
                        running_sum_loss += total_loss.cpu().detach().numpy()
                        running_sum_within_batch_loss += within_batch_loss.cpu().detach().numpy()
                        bar.set_description('loss = {:.6f}, within batch = {:.6f}'.format(running_sum_loss / n, running_sum_within_batch_loss / n))
                    else:
                        running_sum_loss += total_loss.cpu().detach().numpy()
                        bar.set_description('loss = {:.6f}'.format(running_sum_loss / n))
                    if args.debug > 0 and i == args.debug: break

            if mode == 'val':
                with torch.no_grad():
                    all_val_im_embs = torch.cat(all_val_im_embs).cpu()
                    all_val_txt_embs = torch.cat(all_val_txt_embs).cpu()

                im2text_dist = sklearn.metrics.pairwise_distances(all_val_im_embs,
                                                                  all_val_txt_embs,
                                                                  metric='cosine',
                                                                  n_jobs=args.workers_dataloader)

                im2text_ranks = np.diagonal(rankdata(im2text_dist, axis=0))
                text2im_ranks = np.diagonal(rankdata(im2text_dist, axis=1))
                print('im2text rank: {:.1f}, text2im rank: {:.1f}, size: {}'.format(
                    np.mean(im2text_ranks),
                    np.mean(text2im_ranks),
                    len(text2im_ranks)))

                val_loss = running_sum_loss / n if args.val_stat == 'loss' else (np.mean(im2text_ranks) + np.mean(text2im_ranks)) / 2
                reduce_lr.step(val_loss)

                if val_loss < best_val_loss:
                    print('{} is a lower val loss than {}, saving weights!'.format(
                        val_loss,
                        best_val_loss))
                    best_val_loss = val_loss
                    torch.save(
                        {'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optim.state_dict()},
                        tmpfile.name)
                    not_improved_epoch = 0
                else:
                    not_improved_epoch += 1

        if args.early_stop > 0 and not_improved_epoch == args.early_stop:
            print('Havent improved in {} epochs, breaking.'.format(not_improved_epoch))
            break
        if args.debug == 1: break

    state = torch.load(tmpfile.name)
    optim.load_state_dict(state['optimizer_state_dict'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    final_path = args.output_model_path.format(best_val_loss) + '.pt'
    final_path_predictions =  args.output_predictions_path.format(best_val_loss)
    print('saving to {}'.format(final_path))
    print('and predictions from {}'.format(final_path_predictions))
    torch.save(
        {'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optim.state_dict()},
        final_path)


if __name__ == '__main__':
    main()
