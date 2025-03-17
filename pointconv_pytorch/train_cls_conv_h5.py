# 这个是旧的，被2025.03.04丢弃，这个是对应于单个class存储的H5数据，还得映射的
import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_checkpoint
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
import provider
import numpy as np


def count_classes(train_path):
    """
    根据训练集目录统计类别数。
    """
    class_dirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    return len(class_dirs)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    # 修改批次大小为50，即每次epoch训练traindata/batchsize次数，如200数据，batchszie=20，每次epoch要输入10次数据，训练10次
    parser.add_argument('--batchsize', type=int, default=20, help='batch size in training')
    parser.add_argument('--epoch',  default=1, type=int, help='number of epoch in training')  # 修改epoch从400到200或150
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')  # 修改学习率由0.001为0.01
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1200, help='Point Number [default: 1024]')  # 这个点云数量根据自己的修改
    parser.add_argument('--num_workers', type=int, default=16, help='Worker Number [default: 16]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None, help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    # 添加训练和测试文件路径
    parser.add_argument('--train_path', type=str, default='I:/sandprocess/data/originbad/fps_subsets/train',
                        help='Path to training data')
    parser.add_argument('--test_path', type=str, default='I:/sandprocess/data/originbad/fps_subsets/test',
                        help='Path to testing data')
    return parser.parse_args()


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''自动识别类别数'''
    num_classes = count_classes(args.train_path)
    print(f'Detected {num_classes} classes.')
    args.num_classes = num_classes  # 更新参数中的类别数

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)

    file_dir = Path(str(experiment_dir) + f'/{args.model_name}_classes{args.num_classes}_points{args.num_point}_' +
                    str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + f'/train_{args.model_name}_cls.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = os.path.dirname(args.train_path)

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point,
                                       split='train', normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point,
                                      split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batchsize,
                                                  shuffle=True, num_workers=args.num_workers)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize,
                                                 shuffle=False, num_workers=args.num_workers)

    logger.info(f'Training samples: {len(TRAIN_DATASET)}, Testing samples: {len(TEST_DATASET)}')

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    classifier = PointConvClsSsg(args.num_classes).to(device)
    if args.pretrain is not None:
        logger.info('Loading pre-trained model...')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''TRAINING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):

        print(f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):')
        start_time = time.time() # 保留总epoch计时
        logger.info(f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):')
        classifier.train()
        mean_correct = []

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.detach().numpy()

            points[:, :, 0:6] = provider.inject_gaussian_noise(points[:, :, 0:6])  # 注入高斯噪声
            points[:, :, 0:6] = provider.rotate_point_cloud_with_normal(points[:, :, 0:6])  # 随机旋转
            jittered_data = provider.random_scale_point_cloud(points[:, :, 0:3], scale_low=0.95, scale_high=1.05)  # 随机缩放
            jittered_data = provider.shift_point_cloud(jittered_data, shift_range=0.2)  # 随机平移
            points[:, :, 0:3] = jittered_data

            points = provider.random_point_dropout_v2(points)
            points = provider.shuffle_points(points)

            points = torch.Tensor(points).transpose(2, 1).to(device)
            target = target[:, 0].to(device)

            optimizer.zero_grad()
            pred = classifier(points[:, :3, :], points[:, 3:, :])
            loss = F.nll_loss(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()

        train_acc = np.mean(mean_correct)
        print(f'Train Accuracy: {train_acc:.4f}')
        logger.info(f'Train Accuracy: {train_acc:.4f}')

        acc = test(classifier, testDataLoader)

        if (acc >= best_tst_accuracy) and epoch > 5:
            best_tst_accuracy = acc
            logger.info('Saving model...')
            save_checkpoint(global_epoch + 1, train_acc, acc, classifier, optimizer, str(checkpoints_dir), args.model_name)
            print('Saving model....')
        end_time = time.time()
        epoch_time = end_time - start_time  # 原
        completion_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print('\r Loss: %f' % loss.data)
        logger.info('Loss: %.2f', loss.data)
        print('\r Test %s: %f   ***  %s: %f' % (blue('Accuracy'),acc, blue('Best Accuracy'),best_tst_accuracy))
        logger.info('Test Accuracy: %f  *** Best Test Accuracy: %f', acc, best_tst_accuracy)
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s at {completion_time}.")
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s at {completion_time}.")

        global_epoch += 1

    logger.info(f'Training completed. Best Test Accuracy: {best_tst_accuracy:.4f}')

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    print(f'The train dataset path: {args.train_path}' + '\n'+f'The test dataset path: {args.test_path}')
    main(args)
