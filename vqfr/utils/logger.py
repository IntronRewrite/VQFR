import datetime
import logging
import time

from .dist_util import get_dist_info, master_only

initialized_logger = {}


class AvgTimer():

    def __init__(self, window=200):
        self.window = window  # average window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        self.start_time = time.time()

    def record(self):
        self.count += 1
        self.current_time = time.time() - self.start_time
        self.total_time += self.current_time
        # calculate average time
        self.avg_time = self.total_time / self.count
        # reset
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self):
        self.start_time = time.time()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # 记录当前的 epoch 和迭代次数，以及学习率列表
        epoch = log_vars.pop('epoch')  # 从日志变量中取出 epoch
        current_iter = log_vars.pop('iter')  # 从日志变量中取出当前迭代次数
        lrs = log_vars.pop('lrs')  # 从日志变量中取出学习率列表

        # 构建日志信息的初始部分，包括实验名称、epoch、迭代次数和学习率
        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
        for v in lrs:  # 遍历学习率列表
            message += f'{v:.3e},'  # 将学习率格式化为科学计数法并添加到日志信息中
        message += ')] '

        # 如果日志变量中包含时间信息，则计算时间相关的日志信息
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')  # 取出每次迭代的时间
            data_time = log_vars.pop('data_time')  # 取出每次迭代的数据加载时间

            # 计算总耗时、平均每次迭代耗时以及预计剩余时间
            total_time = time.time() - self.start_time  # 从开始时间到现在的总耗时
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)  # 平均每次迭代耗时
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)  # 预计剩余时间
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))  # 将预计剩余时间格式化为可读字符串
            message += f'[eta: {eta_str}, '  # 添加预计剩余时间到日志信息
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '  # 添加迭代时间和数据加载时间到日志信息

        # 遍历日志变量中的其他键值对，通常是损失值等
        for k, v in log_vars.items():
            message += f'{k}: {v:.4e} '  # 将键值对格式化为科学计数法并添加到日志信息中
            # 如果启用了 TensorBoard 日志记录器，并且实验名称中不包含 "debug"
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('l_'):  # 如果键以 'l_' 开头，表示是损失值
                    self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)  # 将损失值记录到 TensorBoard
                else:  # 其他键值对
                    self.tb_logger.add_scalar(k, v, current_iter)  # 将键值对记录到 TensorBoard
        self.logger.info(message)  # 将最终的日志信息输出到日志记录器

        


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = get_root_logger()

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(id=wandb_id, resume=resume, name=opt['name'], config=opt, project=project, sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='vqfr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'vqfr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    msg = r"""
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: ' f'\n\tPyTorch: {torch.__version__}' f'\n\tTorchVision: {torchvision.__version__}')
    return msg
