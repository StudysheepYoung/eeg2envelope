"""
统一的训练日志管理模块
整合TensorBoard日志、终端输出、可视化等功能
"""
import os
import matplotlib.pyplot as plt
import torch


class TrainingLogger:
    """
    统一的训练日志管理类

    功能:
    - TensorBoard标量、直方图、图像记录
    - 终端输出
    - 可视化图像保存
    - 梯度监控
    """

    def __init__(self, writer, save_path, is_main_process, enable_grad_histogram=True):
        """
        初始化日志管理器

        参数:
        writer: TensorBoard SummaryWriter实例
        save_path: 结果保存路径
        is_main_process: 是否为主进程（分布式训练使用）
        enable_grad_histogram: 是否启用梯度直方图记录（默认True）
        """
        self.writer = writer
        self.save_path = save_path
        self.is_main = is_main_process
        self.enable_grad_histogram = enable_grad_histogram
        self.viz_dir = os.path.join(save_path, 'visualizations') if save_path else None

    def print(self, content):
        """统一的终端输出接口"""
        if self.is_main:
            print(content)

    def log_training(self, epoch, total_epochs, step, total_steps, loss_dict, lr, speed, time_remaining, global_step, print_to_terminal=False):
        """
        记录训练信息

        参数:
        epoch: 当前epoch
        total_epochs: 总epoch数
        step: 当前步数
        total_steps: 每epoch的总步数
        loss_dict: 损失字典，如 {'total': 1.5, 'mse': 0.9, 'pearson': 0.6}
        lr: 学习率
        speed: 训练速度（批/秒）
        time_remaining: 剩余时间（秒）
        global_step: 全局步数
        print_to_terminal: 是否打印到终端（默认False，避免与tqdm冲突）
        """
        # 终端输出（使用tqdm时关闭，避免冲突）
        if print_to_terminal:
            self.print(
                f'Epoch:[{epoch}/{total_epochs}]({step}/{total_steps}) '
                f'loss:{loss_dict.get("total", 0):.3f} lr:{lr:.5f} '
                f'速度:{speed:.2f}批/秒 总剩余:{time_remaining:.2f}秒'
            )

        # TensorBoard记录 - 训练指标统一放在 Train/ 下
        if self.writer and self.is_main:
            # 总损失
            self.writer.add_scalar("Train/loss_total", loss_dict.get('total', 0), global_step)

            # 损失组成部分
            if 'mse' in loss_dict:
                self.writer.add_scalar("Train/loss_mse", loss_dict['mse'], global_step)
            if 'pearson' in loss_dict:
                self.writer.add_scalar("Train/loss_pearson", loss_dict['pearson'], global_step)
            if 'l1' in loss_dict:
                self.writer.add_scalar("Train/loss_l1", loss_dict['l1'], global_step)

    def log_validation(self, global_step, loss, metric):
        """
        记录验证信息

        参数:
        global_step: 全局步数
        loss: 验证损失
        metric: 验证指标（如Pearson相关系数）
        """
        # 终端输出
        self.print(f'|-Validation-|Step:{global_step}: loss:{loss:.3f} metric:{metric:.3f}')

        # TensorBoard记录 - 验证指标统一放在 Validation/ 下
        if self.writer and self.is_main:
            self.writer.add_scalar("Validation/loss", loss, global_step)
            self.writer.add_scalar("Validation/pearson", metric, global_step)

    def log_test(self, global_step, loss, metric):
        """
        记录测试信息

        参数:
        global_step: 全局步数
        loss: 测试损失
        metric: 测试指标
        """
        # 终端输出
        self.print(f'|-Test-|Step:{global_step}: loss:{loss:.3f} metric:{metric:.3f}')

        # TensorBoard记录 - 测试指标统一放在 Test/ 下
        if self.writer and self.is_main:
            self.writer.add_scalar("Test/loss", loss, global_step)
            self.writer.add_scalar("Test/pearson", metric, global_step)

    def log_gradients(self, model, global_step, key_layers_only=True):
        """
        记录模型梯度信息

        参数:
        model: 训练模型
        global_step: 全局步数
        key_layers_only: 是否只记录关键层（默认True，减少数据量）
        """
        if not self.writer or not self.is_main:
            return

        # 获取实际模型（处理DDP包装）
        actual_model = model.module if hasattr(model, 'module') else model

        # 计算全局梯度范数（始终记录，这是重要指标）
        total_norm = 0
        for p in actual_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # 记录全局梯度范数 - 梯度指标统一放在 Gradient/ 下
        self.writer.add_scalar("Gradient/norm", total_norm, global_step)

        # 记录直方图（如果启用）
        if self.enable_grad_histogram:
            if key_layers_only:
                # 只记录关键层，减少数据量约90%
                key_patterns = ['layer_stack.0', 'layer_stack.7', 'fc', 'conv1', 'slf_attn']
                for name, param in actual_model.named_parameters():
                    if param.grad is not None:
                        # 检查是否为关键层
                        if any(pattern in name for pattern in key_patterns):
                            self.writer.add_histogram(f"Gradient/{name}", param.grad.data, global_step)
                            self.writer.add_histogram(f"Weight/{name}", param.data, global_step)
            else:
                # 记录所有层（完整模式）
                for name, param in actual_model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"Gradient/{name}", param.grad.data, global_step)
                        self.writer.add_histogram(f"Weight/{name}", param.data, global_step)

    def log_visualization(self, predictions, targets, epoch, global_step, save_png=True):
        """
        统一的可视化接口 - 同时保存到TensorBoard和PNG文件

        参数:
        predictions: 预测值（numpy数组）
        targets: 真实值（numpy数组）
        epoch: 当前epoch
        global_step: 全局步数
        save_png: 是否保存PNG文件（默认True）
        """
        if not self.is_main:
            return

        # 创建可视化图（只创建一次）
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(targets, 'b-', label='real envelop', linewidth=1.5)
        ax.plot(predictions, 'r-', label='rebuild envelop', linewidth=1.5)
        ax.set_title(f'Epoch {epoch}')
        ax.set_xlabel('time')
        ax.set_ylabel('amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 保存到TensorBoard
        if self.writer:
            self.writer.add_figure('语音包络可视化', fig, global_step)

        # 可选：保存PNG文件
        if save_png and self.viz_dir:
            os.makedirs(self.viz_dir, exist_ok=True)
            save_path = os.path.join(self.viz_dir, f'envelope_viz_epoch_{epoch}.png')
            fig.savefig(save_path, dpi=100, bbox_inches='tight')

        # 关闭图像释放内存
        plt.close(fig)

    def log_scalar(self, tag, value, step):
        """通用的标量记录接口"""
        if self.writer and self.is_main:
            self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        """通用的直方图记录接口"""
        if self.writer and self.is_main:
            self.writer.add_histogram(tag, values, step)
