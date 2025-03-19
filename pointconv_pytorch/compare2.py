def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''LOAD DATASET AND OBTAIN NUMBER OF CLASSES'''
    # 如果有checkpoint，加载其中的类别数和标签映射
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        num_classes = checkpoint['num_classes']
        label_map = checkpoint['label_map']
        logger.info("Loaded label_map and num_classes from checkpoint.")
    else:
        # 如果没有checkpoint，加载训练数据后获取类别数和标签映射
        TRAIN_DATASET = ModelNetDataLoader(root=args.train_path, npoint=args.num_point, split='train', normal_channel=args.normal)
        num_classes = len(TRAIN_DATASET.classes)
        label_map = TRAIN_DATASET.label_map  # 获取训练数据的标签映射
        logger.info(f"Detected {num_classes} classes from training data.")

    # 设置 args.num_classes，在后续文件路径创建时使用
    args.num_classes = num_classes

    # 传递给 test 数据加载器
    TEST_DATASET = ModelNetDataLoader(root=args.test_path, npoint=args.num_point, split='test', normal_channel=args.normal, label_map=label_map, num_classes=num_classes)

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)

    # 在这之前已经设置了 args.num_classes
    file_dir = Path(str(experiment_dir) + f'/{args.model_name}_classes{args.num_classes}_points{args.num_point}_' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOGGING'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + f'/train_{args.model_name}_cls.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRAINING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    # 后续代码保持不变...
