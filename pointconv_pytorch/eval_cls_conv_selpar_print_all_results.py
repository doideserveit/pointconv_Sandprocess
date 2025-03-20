# 这个版本是20250313基于源代码修改的验证脚本，读取h5数据的
import argparse
import os
import sys
import numpy as np 
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


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv Eval')
    parser.add_argument('--batchsize', type=int, default=50, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--num_point', type=int, default=1200, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=16, help='Worker Number [default: 16]')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    # 新增数据根目录参数，用于指定存放生成的 h5 数据的目录
    parser.add_argument('--data_root', type=str, default='./data/origin1k', help='Root directory of the h5 data')
    # 新增数据集类型参数，用于指定使用的数据集类型
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'eval'], help='Dataset split to use (train, test, eval)')
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.checkpoint is None:
        print("Please load Checkpoint to eval...")
        sys.exit(0)

    '''CREATE DIR for Evaluation Results'''
    experiment_dir = Path('./eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + f'/{args.model_name}_Eval-' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.checkpoint, checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + f'/eval_{args.model_name}_cls.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = args.data_root
    logger.info(f"Data root: {DATA_PATH}, filename: all_{args.split}_fps_subsets.h5")

    ckpt = torch.load(args.checkpoint, map_location='cuda:0')
    num_class = ckpt['num_classes']
    label_map = ckpt['label_map']
    logger.info("Loaded num_classes and label_map from checkpoint")
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split=args.split, normal_channel=args.normal, label_map=label_map, num_classes=num_class)  
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)
    logger.info("The number of %s data is: %d", args.split, len(TEST_DATASET))

    # 获取反向标签映射，用于将映射后的标签转换回原始标签
    inverse_label_map = {v: k for k, v in TEST_DATASET.label_map.items()}# 获取反向标签映射，用于将映射后的标签转换回原始标签
    logger.info("Inverse label mapping (model output -> original): %s", str(inverse_label_map))

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    classifier = PointConvClsSsg(num_class, label_map).cuda()  # 直接使用checkpoint中信息创建模型
    print('Load CheckPoint...')
    logger.info('Load CheckPoint')
    ckpt_data = torch.load(args.checkpoint, weights_only=False)
    start_epoch = ckpt_data['epoch']
    classifier.load_state_dict(ckpt_data['model_state_dict'])  # 载入checkpoint参数

    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')

    classifier = classifier.eval()
    mean_correct = []
    # 用于记录前几个样本的预测结果和真实标签（反映射回原始标签）
    sample_pred_orig = []
    sample_true_orig = []

    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        pointcloud, target = data
        # target: shape (B, 1) mapped label
        target = target[:, 0]

        points = pointcloud.permute(0, 2, 1)
        points, target = points.cuda(), target.cuda()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

        # 修改: 不再限制只保存前10个样本，将所有样本原始标签保存到 sample_pred_orig 和 sample_true_orig 中
        pred_orig = [inverse_label_map[x.item()] for x in pred_choice]
        target_orig = [inverse_label_map[x.item()] for x in target]
        sample_pred_orig.extend(pred_orig)
        sample_true_orig.extend(target_orig)

    accuracy = np.mean(mean_correct)
    print('Total Accuracy: %f' % accuracy)
    logger.info('Total Accuracy: %f' % accuracy)
    
    # 修改后打印所有样本的预测与真实标签（原始标签）
    print("Sample predictions (original labels):")
    for i in range(len(sample_pred_orig)):  # 修改：打印所有样本而非只打印前10个
        print(f"Sample {i}: Predicted: {sample_pred_orig[i]}, Ground Truth: {sample_true_orig[i]}")
    logger.info("Sample predictions (original labels): %s", str(list(zip(sample_pred_orig, sample_true_orig))))
    logger.info('End of evaluation...')
    print(f'log saved to {log_dir}')

if __name__ == '__main__':
    args = argparse.Namespace(
        batchsize=10,
        gpu='0',
        checkpoint='/share/home/202321008879/experiment/originnew_labbotm1k_classes1000_points1200_2025-03-19_12-38/checkpoints/originnew_labbotm1k-0.995400-0099.pth',
        num_point=1200,
        num_workers=24,
        model_name='originnew_labbotm1k',
        normal=True,
        data_root='./data/h5data/load1_selpar',
        split='eval'  # h5_filename = os.path.join(self.root, f"all_{split}_fps_subsets.h5")
    )
    main(args)
# # 原来的实现中采用了批次平均方式（mean_correct 列表的均值），这会导致当各批次样本数不相等时，不准确地反映整体准确率。
#修改后的实现使用了全局累加的方式，将所有批次的正确样本数和总样本数汇总，然后通过总正确率进行计算，这样避免了批次内部样本数差异带来的误差。
#原因分析： 原来的方法计算每个批次的正确率，然后取均值，当最后一个批次样本数不足时，该批次贡献的权重与其他批次不一致，导致整体准确率偏差。而新的方法直接统计整个测试集的正确个数与总样本数，保证每个样本都有相同比例的贡献，从而得到正确的准确率。
# 所以建议调整batchsize使各批次数量相等