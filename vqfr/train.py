import datetime  # 导入日期时间模块
import logging  # 导入日志模块
import math  # 导入数学模块
import time  # 导入时间模块
import torch  # 导入 PyTorch
from os import path as osp  # 导入 os.path 模块并重命名为 osp

# 导入自定义模块和工具函数
from vqfr.data import build_dataloader, build_dataset  # 数据加载器和数据集构建函数
from vqfr.data.data_sampler import EnlargedSampler  # 数据采样器
from vqfr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher  # 数据预取器
from vqfr.models import build_model  # 模型构建函数
from vqfr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                        init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)  # 工具函数
from vqfr.utils.options import copy_opt_file, dict2str, parse_options  # 配置文件相关工具函数


def init_tb_loggers(opt):
    # 初始化 TensorBoard 和 WandB 日志记录器
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('newquant' not in opt['name']):
        # 如果使用 WandB，则必须启用 TensorBoard
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)  # 初始化 WandB 日志记录器
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'newquant' not in opt['name']:
        # 如果启用了 TensorBoard 日志记录器，则初始化
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # 创建训练和验证数据加载器
    train_loader, val_loaders = None, []  # 初始化训练和验证加载器
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':  # 如果是训练阶段
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)  # 数据集扩展比例
            train_set = build_dataset(dataset_opt)  # 构建训练数据集
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)  # 数据采样器
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])  # 构建训练数据加载器

            # 计算每个 epoch 的迭代次数和总 epoch 数
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])  # 总迭代次数
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))  # 总 epoch 数
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')  # 打印训练统计信息
        elif phase.split('_')[0] == 'val':  # 如果是验证阶段
            val_set = build_dataset(dataset_opt)  # 构建验证数据集
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])  # 构建验证数据加载器
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')  # 打印验证数据集信息
            val_loaders.append(val_loader)  # 添加到验证加载器列表
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')  # 未知阶段抛出异常

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters  # 返回加载器和统计信息


def load_resume_state(opt):
    # 加载恢复状态
    resume_state_path = None
    if opt['auto_resume']:  # 如果启用了自动恢复
        state_path = osp.join('experiments', opt['name'], 'training_states')  # 状态文件路径
        if osp.isdir(state_path):  # 如果路径存在
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))  # 扫描状态文件
            if len(states) != 0:  # 如果存在状态文件
                states = [float(v.split('.state')[0]) for v in states]  # 提取状态文件编号
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')  # 获取最新状态文件路径
                opt['path']['resume_state'] = resume_state_path  # 更新配置
    else:
        if opt['path'].get('resume_state'):  # 如果手动指定了恢复状态路径
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:  # 如果没有恢复状态
        resume_state = None
    else:
        device_id = torch.cuda.current_device()  # 获取当前 GPU 设备 ID
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))  # 加载状态文件
        check_resume(opt, resume_state['iter'])  # 检查恢复状态
    return resume_state  # 返回恢复状态


def train_pipeline(root_path):
    # 训练主流程
    opt, args = parse_options(root_path, is_train=True)  # 解析配置文件
    opt['root_path'] = root_path  # 设置根路径

    torch.backends.cudnn.benchmark = True  # 启用 cuDNN 加速
    # torch.backends.cudnn.deterministic = True  # 可选：设置为确定性模式

    resume_state = load_resume_state(opt)  # 加载恢复状态
    if resume_state is None:  # 如果没有恢复状态
        make_exp_dirs(opt)  # 创建实验目录
        if opt['logger'].get('use_tb_logger') and 'newquant' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))  # 创建 TensorBoard 日志目录

    copy_opt_file(args.opt, opt['path']['experiments_root'])  # 复制配置文件到实验目录

    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")  # 日志文件路径
    logger = get_root_logger(logger_name='vqfr', log_level=logging.INFO, log_file=log_file)  # 初始化日志记录器
    logger.info(get_env_info())  # 打印环境信息
    logger.info(dict2str(opt))  # 打印配置文件内容
    tb_logger = init_tb_loggers(opt)  # 初始化日志记录器

    result = create_train_val_dataloader(opt, logger)  # 创建数据加载器
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result  # 解包结果

    model = build_model(opt)  # 构建模型
    if resume_state:  # 如果有恢复状态
        model.resume_training(resume_state)  # 恢复训练状态
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")  # 打印恢复信息
        start_epoch = resume_state['epoch']  # 恢复起始 epoch
        current_iter = resume_state['iter']  # 恢复起始迭代次数
    else:
        start_epoch = 0  # 起始 epoch 为 0
        current_iter = 0  # 起始迭代次数为 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)  # 创建消息日志记录器

    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')  # 获取数据预取模式
    if prefetch_mode is None or prefetch_mode == 'cpu':  # 如果是 CPU 预取模式
        prefetcher = CPUPrefetcher(train_loader)  # 使用 CPU 预取器
    elif prefetch_mode == 'cuda':  # 如果是 CUDA 预取模式
        prefetcher = CUDAPrefetcher(train_loader, opt)  # 使用 CUDA 预取器
        logger.info(f'Use {prefetch_mode} prefetch dataloader')  # 打印预取模式
        if opt['datasets']['train'].get('pin_memory') is not True:  # 如果未启用 pin_memory
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')  # 抛出异常
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")  # 抛出异常

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')  # 打印训练开始信息
    data_timer, iter_timer = AvgTimer(), AvgTimer()  # 初始化计时器
    start_time = time.time()  # 记录开始时间

    for epoch in range(start_epoch, total_epochs + 1):  # 遍历每个 epoch
        train_sampler.set_epoch(epoch)  # 设置采样器的 epoch
        prefetcher.reset()  # 重置预取器
        train_data = prefetcher.next()  # 获取第一批数据

        while train_data is not None:  # 遍历每批数据
            data_timer.record()  # 记录数据加载时间

            current_iter += 1  # 更新当前迭代次数
            if current_iter > total_iters:  # 如果超过总迭代次数
                break
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))  # 更新学习率
            model.feed_data(train_data)  # 将数据输入模型
            model.optimize_parameters(current_iter)  # 优化模型参数
            iter_timer.record()  # 记录迭代时间
            if current_iter == 1:  # 如果是第一次迭代
                msg_logger.reset_start_time()  # 重置消息日志记录器的开始时间
            if current_iter % opt['logger']['print_freq'] == 0:  # 如果到达打印频率
                log_vars = {'epoch': epoch, 'iter': current_iter}  # 日志变量
                log_vars.update({'lrs': model.get_current_learning_rate()})  # 添加学习率
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})  # 添加时间信息
                log_vars.update(model.get_current_log())  # 添加模型日志
                msg_logger(log_vars)  # 打印日志

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:  # 如果到达保存频率
                logger.info('Saving models and training states.')  # 打印保存信息
                model.save(epoch, current_iter)  # 保存模型和状态

            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):  # 如果需要验证
                if len(val_loaders) > 1:  # 如果有多个验证数据集
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')  # 打印警告

                if len(val_loaders) == 0:  # 如果没有验证数据集
                    model.validation(None, current_iter, tb_logger, opt['val']['save_img'])  # 执行验证
                else:
                    for val_loader in val_loaders:  # 遍历验证数据集
                        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])  # 执行验证

            data_timer.start()  # 开始数据计时
            iter_timer.start()  # 开始迭代计时
            train_data = prefetcher.next()  # 获取下一批数据

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))  # 计算总耗时
    logger.info(f'End of training. Time consumed: {consumed_time}')  # 打印训练结束信息
    logger.info('Save the latest model.')  # 打印保存最新模型信息
    model.save(epoch=-1, current_iter=-1)  # 保存最新模型
    if opt.get('val') is not None:  # 如果需要验证
        for val_loader in val_loaders:  # 遍历验证数据集
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])  # 执行验证
    if tb_logger:  # 如果启用了 TensorBoard
        tb_logger.close()  # 关闭 TensorBoard 日志记录器


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))  # 获取项目根路径
    train_pipeline(root_path)  # 启动训练流程
