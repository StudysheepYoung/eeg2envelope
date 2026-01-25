# NeuroConformer 脚本运行说明

下表覆盖仓库中所有 `.py` 脚本，对每个脚本给出功能、主要输入(命令行参数/依赖数据)及输出(文件或副产物)。按用途分组，查找特定脚本时可按标题定位。

## 训练与消融脚本

### train.py
- **功能**：改进版 Conformer 训练入口，支持 CNN/SE 开关、门控残差、MLP 输出头、LLRD、AMP、DDP 等，用于 EEG→语音包络回归。
- **输入**：命令行超参（如 `--epoch --batch_size --win_len --gpu --use_ddp --no_skip_cnn --no_se --experiment_folder` 等），`--dataset_folder`/`--split_folder` 指向的数据分片及 `data/<split>/train_-_*`/`val_-_*` `.npy` 文件。
- **输出**：`test_results/<experiment>/best_model.pt` 与 `checkpoint_*`、`tb_logs`、日志/可视化 PNG，TensorBoard events。

### test_model.py
- **功能**：加载 checkpoint 并在测试集 (1-85 受试者) 评估 Pearson、绘制箱线图，可批量评测。
- **输入**：`--checkpoint` 或 `--checkpoint_dir + --pattern`，`--split_data_dir`、`--test_data_dir`，可选 `--save_predictions`、`--gpu`。
- **输出**：`test_results_eval/<checkpoint_name>/test_results.json`、`pearson_boxplot_1_71.png`、对比 CSV、(可选) `predictions/*.npy`。

### ablation_inference.py
- **功能**：根据实验别名或路径批量载入 `best_model.pt` 做推理，保存每个实验的受试者 Pearson 统计与汇总。
- **输入**：`--models <Exp别名...>` (默认映射已在脚本中定义)，`--output_dir`，`--gpu`，数据目录复用 `test_model`。
- **输出**：`ablation_results/<Exp>_results.json` (单模型) 与 `ablation_results/ablation_all_results.json`（含 1-71/72-85 平均值、原始分布）。

### ablation_experiment.py
- **功能**：快速指定一个或多个 checkpoint 文件夹/别名并评测，直接绘制箱线图与比较表。
- **输入**：`--model/--models` 或 `--folder/--folders`，别名管理参数(`--add-alias/--list-aliases`)，`--split_data_dir`、`--test_data_dir`、`--output_dir`、`--gpu`。
- **输出**：`ablation_results/all_models_results.json`、`comparison.csv`、`all_models_comparison_boxplot.png`。

### ablation_plot.py
- **功能**：从 `ablation_results` 读取推理结果，生成统一箱线图、比较/配置表，支持数值微调与模型筛选。
- **输入**：`--results_dir`、`--output_dir`、`--adjust "Exp-XX:+0.02"`、`--select "Exp-00,..."`。
- **输出**：`ablation_plots/ablation_boxplot.png`、`comparison_table.csv`、`model_config_table.csv` 等。

### ablation_plot_violin.py
- **功能**：同上但输出小提琴图，并按照受试者画连线对比跨实验表现。
- **输入**：同 `ablation_plot.py`。
- **输出**：`ablation_plots/ablation_violin_with_lines.png` 与日志。

## 评估、绘图与比较脚本

### compare_all_models.py
- **功能**：汇总 ADT 基线与 NeuroConformer `test_results.json`，绘制箱线图并做成对统计检验。
- **输入**：脚本内硬编码的 baseline/Conformer JSON 路径（需存在），运行时无必填 CLI。
- **输出**：`comparison_results/all_models_comparison.png`、`comparison_summary.csv`、`statistical_tests.csv`。

### plot_tensorboard.py
- **功能**：读取单个 TensorBoard `events` 文件中的标量，绘制指定步数区间的原始曲线。
- **输入**：`--log_dir`、`--scalar`、范围/样式参数 (`--step_min/max --color --linewidth --no_grid` 等)。
- **输出**：默认 `tensorboard_curve.png`（可用 `--output` 指定路径）。

### plot_cross_subject_analysis.py
- **功能**：根据 `test_results.json` 生成 1-71 vs 72-85 受试者的 CDF/箱线图；支持 `--all_models`（遍历比较脚本中的模型）或 `--ablation`（读取消融结果）。
- **输入**：`--json_path`、`--output_dir`、`--all_models`、`--ablation --ablation_dir --grouped` 等。
- **输出**：`cross_subject_analysis/` 下的 `cdf_trainset_only.png`、分组 CDF/箱线图以及相应 CSV。

### plot_prediction_quality.py
- **功能**：对单个 `test_results.json` 生成多种质量图：时序对比、误差分布、散点、受试者相关性、时间窗口分析。
- **输入**：`--json_path`、`--output_dir`。
- **输出**：`prediction_analysis/` 中的 `time_series_comparison.png`、`error_distribution.png`、`scatter.png`、`subject_distribution.png` 等。

### plot_params_vs_pearson.py
- **功能**：基于脚本内的论文数据绘制“参数量 vs Pearson”散点/组合图，可用于展示模型效率。
- **输入**：无 CLI；如需修改请直接编辑脚本。
- **输出**：`comparison_results/params_vs_pearson.png`、`params_vs_pearson.pdf`、组合图/表格文件。

### extract_se_channel_importance.py
- **功能**：对 `skip_cnn=True且use_se=True` 的 checkpoint 提取 SE 通道注意力，反投影成 64 通道权重并做脑区统计。
- **输入**：`--checkpoint`、`--split_data_dir`、`--output_dir`、`--gpu`、`--batch_size`、可选 `--save_per_sample`、`--no_visualize`。
- **输出**：`se_channel_analysis/<exp>/` 下包含 raw SE 权重、64d 权重 `.npy`、脑区 `json`、统计表、可视化 PNG。

## 模型定义脚本

### models/FFT_block_conformer_v2.py
- **功能**：主力 Decoder（门控残差、可选 CNN/SE/MLP 头、梯度缩放等），供训练/推理脚本导入。
- **输入**：构造参数如 `in_channel, d_model, n_layers, use_gated_residual, skip_cnn` 等；`forward(eeg, sub_id)` 输入 `[B, 64, T]` EEG 与受试者 ID。
- **输出**：预测的 `[B, T, 1]` 包络及中间模块供日志/Hook 使用。

### models/FFT_block_conformer.py
- **功能**：早期 Conformer Decoder（固定三层 CNN+SE），支持可选正弦位置编码。
- **输入**：同上但无门控/MLP 相关参数；`forward` 接受 EEG 张量与 subject id。
- **输出**：`[B, T, 1]` 预测值。

### models/FFT_block.py
- **功能**：Transformer 版本 Decoder，三层 CNN 后接 Pre-LN Transformer Block，含 SE 通道注意力。
- **输入/输出**：同上。

### models/FFT_block_initial_.py
- **功能**：最初 FFT Block（单卷积+SE+Transformer），用于复现最早实验。
- **输入/输出**：同上。

### models/ConformerLayers.py
- **功能**：实现 Conformer 组件（Swish、GLU、卷积模块、相对位置编码、ConformerBlock）。
- **输入**：构造时指定 `d_model, n_head, kernel_size, dropout` 等；`forward` 接受 `[B, T, d_model]`。
- **输出**：增强后的 `[B, T, d_model]` 特征。

### models/SubLayers.py
- **功能**：Transformer 子层（多头注意力、前馈、Macaron FFN 等），供旧模型/辅助模块使用。
- **输入**：`MultiHeadAttention/PositionwiseFeedForward` 的维度配置与 `q,k,v` 张量。
- **输出**：注意力输出、残差堆叠结果。

### models/__init__.py
- **功能**：声明 `models` 包；无额外输入输出。

## 数据与工具脚本

### util/dataset.py
- **功能**：`RegressionDataset` 预加载 EEG/Envelope，支持训练阶段多窗口随机采样、测试阶段整段切片。
- **输入**：文件路径列表、`input_length`、通道数、task 名称、`windows_per_sample`。
- **输出**：`__getitem__` 返回 `(EEG窗口, Envelope窗口, subject_id)`，供 `DataLoader` 使用。

### util/cal_pearson.py
- **功能**：Pearson/L1/MSE、多尺度 Pearson、方差比、SI-SDR 等损失/指标实现。
- **输入**：`y_true`、`y_pred` 张量与可选参数。
- **输出**：逐批损失或指标值供训练/评估调用。

### util/logger.py
- **功能**：`TrainingLogger` 统一管理 TensorBoard/终端日志、梯度直方图、可视化。
- **输入**：初始化参数 (`writer, save_path, is_main_process`)；运行时传入损失、预测、模型等。
- **输出**：TB 标量/直方图、`visualizations/*.png`、控制台信息。

### util/utils.py
- **功能**：通用工具函数：`CustomWriter`、`save_checkpoint`、默认训练解析器。
- **输入**：日志目录、模型/优化器状态、命令行参数。
- **输出**：SummaryWriter 实例、checkpoint 文件、`argparse.ArgumentParser`。

### util/testing_utils.py
- **功能**：加载 `split_data` 下测试样本、将序列切成固定长度窗口。
- **输入**：`split_data_dir`、`sample_rate`、`win_len`，或任意 `np.ndarray`。
- **输出**：`[(eeg, envelope, sub_id)]` 列表、`segment_data` 输出的窗口列表。

### util/biosemi_64_layout.py
- **功能**：定义 BioSemi-64 通道顺序与脑区映射，供 SE 通道分析/可视化使用。
- **输入/输出**：导入后可直接访问 `BIOSEMI_64_CHANNELS`、`BRAIN_REGIONS`。

### util/__init__.py
- **功能**：声明 `util` 包；无额外输入输出。

