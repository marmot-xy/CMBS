import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from model.main_model import weak_main_model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset_weak import AVEDataset

# =================================  seed config ============================
SEED = 666
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
config_path = 'configs/weak.json'
with open(config_path) as fp:
    config = json.load(fp)
print(config)

def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving model '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in {args.snapshot_pref}/Eval.log.')

    '''Dataset'''
    train_dataloader = DataLoader(
        AVEDataset('./data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        AVEDataset('./data/', split='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    '''model setting'''
    mainModel = main_model(config["model"])
    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=40)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.5)
    criterion = nn.BCEWithLogitsLoss().cuda()
    # criterion_event = nn.CrossEntropyLoss().cuda()
    criterion_event = nn.MultiLabelSoftMarginLoss().cuda()

    '''Resume from a checkpoint'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    '''Only Evaluate'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)

    '''Training and Testing'''
    for epoch in range(args.n_epoch):
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='WeaklySupervised',
                    epoch=epoch + 1,
                )
            print("-----------------------------")
            print("best acc and epoch:", best_accuracy, best_accuracy_epoch)
            print("-----------------------------")
        scheduler.step()


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    model.double()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data

        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()
        is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature)
        # is_event_scores = is_event_scores.transpose(1, 0).squeeze().contiguous()

        import numpy
        # labels_foreground = labels[:, :, :-1]
        # labels_BCE, labels_evn = labels_foreground.max(-1)
        # labels_event, _ = labels_evn.max(-1)
        # loss_is_event = criterion(is_event_scores, labels_BCE.double().cuda())

        loss_event_class = criterion_event(event_scores, labels)
        # loss_av_score = criterion_event(av_score, labels.double())
        loss = loss_event_class
        # loss = loss_event_class + loss_av_score
        loss.backward()

        '''Compute Accuracy'''
        # acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        acc = torch.tensor([0])
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #     logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        # if n_iter % args.print_freq == 0:
        #     logger.info(
        #         f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
        #         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #         f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #         f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})'
        #     )

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg


@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    model.double()

    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.cuda()
        bs = visual_feature.size(0)
        is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature)
        # is_event_scores = is_event_scores.transpose(1, 0).squeeze()
        #
        # labels_foreground = labels[:, :, :-1]
        # labels_BCE, labels_evn = labels_foreground.max(-1)
        # labels_event, _ = labels_evn.max(-1)
        # loss_is_event = criterion(is_event_scores, labels_BCE.double().cuda())

        # acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        # raw_logits = raw_logits + 0.5*av_score
        acc = compute_accuracy_supervised(is_event_scores, raw_logits, labels)
        accuracy.update(acc.item(), bs)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Print logs in Terminal'''
        # if n_iter % args.print_freq == 0:
        #     logger.info(
        #         f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\t'
        #         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #         f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #         f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
        #     )

    if not eval_only:
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    logger.info(
        f'**************************************************************************\t'
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
    )

    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    # labels = labels[:, :, :-1]  # 28 denote background
    _, targets = labels.max(-1)
    # pos pred
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0
    # bg_mask = scores_mask * 28 # 28 denotes bg
    _, event_class = event_scores.max(-1)  # foreground classification
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]
    # add mask
    pred[scores_mask] = 28
    # pred += bg_mask
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return acc


def compute_accuracy_weak(event_scores, labels):
    # event_scores: [batch, 10, 29]
    _, pred = event_scores.max(-1)
    pred = pred.transpose(1, 0).contiguous()
    _, target = labels.max(-1)
    correct = pred.eq(target)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return acc


def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':
    main()
