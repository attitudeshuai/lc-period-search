# 难点与测试对应

## 难点 1：不是单条正弦，谐波要一起拟合

推理轴：领域知识（变星光变曲线的傅里叶分解）

落在推理路径上的哪里：变星的光变曲线是周期函数，标准做法是把它按相位展开成常数项加若干阶谐波；脉冲不对称（亮起来快、暗下去慢）时，只留基波会让周期锁定到一半周期上，形状也完全对不上。原实现只解常数项加一条正弦，谐波整块丢掉。加权也不同：每个观测点各有自己的不确定度，权重必须取 1/sigma²，忽略权重会让噪声大的点把周期拽偏。

对应测试：test_half_period_harmonic_rejected、test_period_matches_independent_search

## 难点 2：周期是搜出来的，还要细化

推理轴：数值方法

落在推理路径上的哪里：相位折叠对频率是高度震荡的目标函数，粗步长定到哪个峰就是哪个峰；步长按观测跨度取 1/(4·跨度)，扫完还要在最优点 ±一步之内用黄金分割细化，否则报出来的周期只能准到步长量级。长周期星的频率贴着搜索区间的下边界（0.0208 对 f_min=0.02），循环的边界条件写错就整段错过。

对应测试：test_period_matches_independent_search、test_long_period_near_low_frequency_edge、test_period_uncertainty_from_curvature

## 难点 3：宇宙线尖峰要先剔掉

推理轴：数据口径

落在推理路径上的哪里：观测点里混着宇宙线打出的尖峰（8 倍 sigma 以上），留着会把周期和振幅一起拽偏。剔除口径是按标准化残差判：|(dmag - 模型)/sigma| 超过 sigma_clip 的整点剔除后重拟合，最多剔 max_clip_iter 轮；剔除后还要检查有效点是否够 min_used。原实现一个点都不剔。

对应测试：test_outliers_clipped_before_fit、test_second_star_other_period

## 难点 4：极大时刻、自由度、残差与次优候选

推理轴：业务规则

落在推理路径上的哪里：epoch_days 是基波极大时刻、要落在 [0, P) 内，等于 atan2(b1, a1) 折回相位再换算成时间；半振幅是 sqrt(a1²+b1²)；自由度是 n_used 减掉 2H+1 个参数，reduced_chi2 用 chi2/dof；次优候选周期要在粗扫网格上排除最优频率 ±5% 的邻域后取残差平方和最小的那一个，用来判断这次定出的周期有没有被谐波或日周期缺测的假峰抢走。位数与 meta.model 都有约定。原实现这几栏全没有。

对应测试：test_epoch_and_amplitude_match、test_chi2_and_dof_contract、test_next_candidate_is_reported、test_report_schema_and_rounding

## 难点 5：夜间账本要把跨运行状态维护对

推理轴：跨运行状态（账本合并）

落在推理路径上的哪里：每次复核要把本次记录覆盖到的每一夜写进账本：夜号取 floor(t)，三栏分别是该夜观测点数、留在最终拟合里的点数、保留点标准化残差绝对值的中位数。账本要与旧账本按夜号合并：同夜号整条替换、别的夜原样留着、整份按夜号升序、最多留 64 夜（超出丢夜号最小的）。读不出来或结构不对要当空账本重写，非法记录与账本不可写都要保持原样。这一轴跨运行，单看一份记录写不出来。

对应测试：test_ledger_records_every_night_sorted、test_ledger_replaces_matching_night_in_place、test_ledger_capped_at_64_nights、test_corrupt_ledger_rebuilt

## 跨运行状态（重复运行与不污染）

同一份时序重复出结论，报告逐字节一致（不能写时间戳）；运行失败时不能覆盖上一次的报告，账本也不许动：先写临时文件再原子替换，非法记录以退出码 2 收场且不落文件。对应测试：test_rerun_produces_identical_bytes、test_failed_run_keeps_previous_report、test_failed_run_leaves_ledger_untouched。

非法输入另有 test_invalid_data_rejected_without_output 单独覆盖（观测点不足、时间不单调、sigma 非正、频率范围颠倒、sigma_clip 非正、剔除后有效点不足），边界值另有 test_zero_sigma_clip_rejected、test_zero_f_min_rejected。

## 权重口径

19 条计分测试 + 1 条门禁（test_invalid_data_rejected_without_output，不通过则整份记 0）：周期与独立复算 0.115、谐波判别 0.09、极大时刻与半振幅 0.065、离群点剔除 0.075、第二颗星 0.065、贴边界的长周期 0.065、chi2 与 dof 口径 0.045、周期不确定度 0.05、次优候选 0.05、输出契约 0.025、重复运行 0.025、失败不覆盖 0.02、账本逐夜记录 0.07、账本同夜号替换 0.06、账本容量 0.045、坏账本重写 0.035、非法记录不动账本 0.04、sigma_clip 边界 0.035、f_min 边界 0.025，和为 1。

## 定标依据（自检用）

判分侧独立复算用更细的频率步长（1/(5·跨度)）、自己的黄金分割与剔点循环，同一份曲线与参考解一致到 0.1% 内；账本一栏按参考解给出的保留点集独立算逐夜中位数，与参考解的中位数一致到 0.01 内。本地实测参考解 1.000000、缺陷版 0.000000；缺陷版只解单条正弦、不剔离群点、不做频率细化、不写账本、报告带时间戳且缺 epoch/自由度几栏，20 条全挂。
