import os
import sys
import numpy as np
import argparse
import pprint
import time
import cv2
import torch
from torch.autograd import Variable
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.resnet import resnet


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='target_dataset',
                        type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large image scale',
                        action='store_true')
    parser.add_argument('--xls', dest='extra_large_scale',
                        help='whether use extra large image scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    overallstarttime = time.time()

    args = parse_args()

    print('Called with args:')
    print(args)

    args.imdb_name = args.dataset
    args.imdbval_name = args.dataset

    if args.extra_large_scale:
        args.cfg_file = "cfgs/{}_xls.yml".format(args.net)
    elif args.large_scale:
        args.cfg_file = "cfgs/{}_ls.yml".format(args.net)
    else:
        "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    try:
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs)
    except Exception as err:
        print("training saw Exception:" + str(err))

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    #input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    #if not os.path.exists(input_dir):
    #    raise Exception('There is no input directory for loading network from ' + input_dir)
    #load_name = os.path.join(input_dir,
    #                         'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    load_name = "/home/ubuntu/Dev/git/JRGEMCP_AILAB/faster-rcnn-private/models/res101/coco_dataset_bloomberg_training_[1440]_R_[0.5,_1,_2]_S_[4,_8,_16,_32]_1634490333.1591375/faster_rcnn_1_10_1081.pth"

    # initialize the network here.
    if args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        raise Exception("network is not defined")

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % load_name)
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')
    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):

        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r'.format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            single_test_image = os.path.join(output_dir, 'example_result.png')
            cv2.imwrite(single_test_image, im2show)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))
