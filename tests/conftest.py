# -*- coding: utf-8 -*-
"""pytest 汇总钩子：按权重把每条测试的结果折算成加权分，写进 weighted.txt。

口径：每条测试占一个权重，权重和 = 1（见 WEIGHTS）；只有 PASS 才拿到该条权重；
不在 WEIGHTS 里的测试名一律不给分（防止误加测试把总分抬高）。
这里只写中间结果 weighted.txt（本地调试可用 CT_WEIGHTED_PATH 覆盖），
reward.txt 由 tests/test.sh 在确认 pytest 结构化报告非空之后再落。
"""
import os

WEIGHTS = {
    "test_period_matches_independent_search": 0.14,
    "test_half_period_harmonic_rejected": 0.11,
    "test_epoch_and_amplitude_match": 0.10,
    "test_outliers_clipped_before_fit": 0.10,
    "test_second_star_other_period": 0.09,
    "test_long_period_near_low_frequency_edge": 0.10,
    "test_chi2_and_dof_contract": 0.07,
    "test_invalid_data_rejected_without_output": 0.10,
    "test_report_schema_and_rounding": 0.07,
    "test_rerun_produces_identical_bytes": 0.06,
    "test_failed_run_keeps_previous_report": 0.06,
}

_RESULTS = {}


def pytest_runtest_logreport(report):
    if report.when in ("setup", "call") and report.failed:
        _RESULTS[report.nodeid.split("::")[-1]] = False
    elif report.when == "call" and report.passed:
        _RESULTS.setdefault(report.nodeid.split("::")[-1], True)


def pytest_sessionfinish(session, exitstatus):
    # 完成报告校验：必须有每一条测试的结果，缺任何一条都按 INCOMPLETE 记 0
    missing = [n for n in WEIGHTS if n not in _RESULTS]
    total = 0.0
    if missing:
        print("[INCOMPLETE] 这些测试没有跑完：%s；记 0" % ", ".join(missing))
    else:
        for name, weight in WEIGHTS.items():
            if _RESULTS.get(name) is True:
                total += weight
    path = os.environ.get("CT_WEIGHTED_PATH", "/logs/verifier/weighted.txt")
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("%.6f\n" % total)
