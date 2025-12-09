# /home/nl/disk_8T/lys/Time-LLM/utils/experiment_logger.py
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import sys


class ExperimentLogger:
    """
    实验日志管理器
    为每次训练/测试创建独立的日志文件夹，包含：
    - config.json: 实验配置参数
    - train.log: 训练过程日志
    - results.json: 最终实验结果
    """
    
    def __init__(
        self,
        base_log_dir: str = './logs',
        experiment_name: str = 'experiment',
        args: Optional[Any] = None,
        accelerator: Optional[Any] = None
    ):
        self.accelerator = accelerator
        self.is_main_process = accelerator is None or accelerator.is_local_main_process
        
        # 创建实验唯一标识
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        
        # 创建实验专属文件夹
        self.log_dir = os.path.join(base_log_dir, self.experiment_id)
        if self.is_main_process:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置日志记录器
        self.logger = self._setup_logger()
        
        # 保存配置
        if args is not None and self.is_main_process:
            self.save_config(args)
        
        # 用于收集训练过程中的指标
        self.metrics_history = {
            'train_loss': [],
            'vali_loss': [],
            'test_loss': [],
            'mae_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志记录器"""
        logger = logging.getLogger(self.experiment_id)
        logger.setLevel(logging.INFO)
        logger.handlers = []  # 清除已有handlers
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if self.is_main_process:
            # 文件handler
            log_file = os.path.join(self.log_dir, 'train.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def save_config(self, args) -> None:
        """保存实验配置到JSON文件"""
        config_path = os.path.join(self.log_dir, 'config.json')
        
        # 将args转换为可序列化的字典
        if hasattr(args, '__dict__'):
            config = vars(args).copy()
        else:
            config = dict(args)
        
        # 移除不可序列化的对象
        keys_to_remove = []
        for key, value in config.items():
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            config[key] = str(config[key])
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        self.info(f"Config saved to {config_path}")
    
    def info(self, message: str) -> None:
        """记录INFO级别日志"""
        if self.is_main_process:
            self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """记录WARNING级别日志"""
        if self.is_main_process:
            self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """记录ERROR级别日志"""
        if self.is_main_process:
            self.logger.error(message)
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        vali_loss: float,
        test_loss: float,
        mae_loss: float = None,
        epoch_time: float = None,
        learning_rate: float = None,
        extra_metrics: Dict[str, float] = None
    ) -> None:
        """记录每个epoch的训练结果"""
        # 保存到历史记录
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['vali_loss'].append(vali_loss)
        self.metrics_history['test_loss'].append(test_loss)
        if mae_loss is not None:
            self.metrics_history['mae_loss'].append(mae_loss)
        if epoch_time is not None:
            self.metrics_history['epoch_time'].append(epoch_time)
        if learning_rate is not None:
            self.metrics_history['learning_rate'].append(learning_rate)
        
        # 格式化日志消息
        msg = f"Epoch {epoch:03d} | Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f}"
        if mae_loss is not None:
            msg += f" | MAE: {mae_loss:.7f}"
        if epoch_time is not None:
            msg += f" | Time: {epoch_time:.2f}s"
        if learning_rate is not None:
            msg += f" | LR: {learning_rate:.2e}"
        
        if extra_metrics:
            for key, value in extra_metrics.items():
                msg += f" | {key}: {value:.7f}"
        
        self.info(msg)
    
    def log_iteration(
        self,
        epoch: int,
        iteration: int,
        total_iterations: int,
        loss: float,
        speed: float = None,
        left_time: float = None
    ) -> None:
        """记录迭代过程中的信息"""
        msg = f"Epoch {epoch:03d} | Iter {iteration:05d}/{total_iterations} | Loss: {loss:.7f}"
        if speed is not None:
            msg += f" | Speed: {speed:.4f}s/iter"
        if left_time is not None:
            hours = left_time // 3600
            minutes = (left_time % 3600) // 60
            msg += f" | ETA: {int(hours)}h {int(minutes)}m"
        
        self.info(msg)
    
    def save_results(
        self,
        final_metrics: Dict[str, Any] = None,
        best_epoch: int = None,
        early_stop_epoch: int = None
    ) -> None:
        """保存最终实验结果"""
        if not self.is_main_process:
            return
        
        results = {
            'experiment_id': self.experiment_id,
            'log_dir': self.log_dir,
            'metrics_history': self.metrics_history,
            'best_epoch': best_epoch,
            'early_stop_epoch': early_stop_epoch,
            'final_metrics': final_metrics or {}
        }
        
        # 计算最优指标
        if self.metrics_history['vali_loss']:
            best_vali_idx = self.metrics_history['vali_loss'].index(
                min(self.metrics_history['vali_loss'])
            )
            results['best_results'] = {
                'best_vali_loss': self.metrics_history['vali_loss'][best_vali_idx],
                'best_test_loss': self.metrics_history['test_loss'][best_vali_idx] 
                    if best_vali_idx < len(self.metrics_history['test_loss']) else None,
                'best_mae_loss': self.metrics_history['mae_loss'][best_vali_idx]
                    if best_vali_idx < len(self.metrics_history['mae_loss']) else None,
            }
        
        results_path = os.path.join(self.log_dir, 'results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        self.info(f"Results saved to {results_path}")
        self.info(f"Experiment logs directory: {self.log_dir}")
    
    def get_log_dir(self) -> str:
        """返回日志目录路径"""
        return self.log_dir


def setup_experiment_logger(
    args,
    accelerator=None,
    base_log_dir: str = './logs'
) -> ExperimentLogger:
    """
    快速创建实验日志管理器的辅助函数
    
    Args:
        args: 命令行参数
        accelerator: Accelerator实例（用于分布式训练）
        base_log_dir: 日志根目录
    
    Returns:
        ExperimentLogger实例
    """
    # 构建实验名称
    experiment_name = f"{args.model}_{args.data}_{args.model_comment}"
    
    return ExperimentLogger(
        base_log_dir=base_log_dir,
        experiment_name=experiment_name,
        args=args,
        accelerator=accelerator
    )