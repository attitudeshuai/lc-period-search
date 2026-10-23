# -*- coding: utf-8 -*-
"""变星光变曲线周期复核的验收测试。

每条测试都自己造曲线、跑被测程序、再独立复算一遍期望值，断言全写在本函数内。
权重表放在 tests/conftest.py，score 只按权重汇总。
"""
import json
import math
import os
import subprocess
import sys
import tempfile

APP = os.environ.get("CT_APP_DIR", "/app")
ENTRY = os.path.join(APP, "src", "period.py")
TWO_PI = 2.0 * math.pi


class _Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_curve(period=1.8539, span_days=120.0, n_nights=40, per_night=10,
               sigma=20.0, harmonics=((1, 42.0, 11.0), (2, -14.0, 7.0), (3, 5.0, -3.5)),
               n_spikes=9, n_harm=3, seed=771003):
    """造一条只按夜窗口采样、带宇宙线尖峰的变星光变曲线。"""
    rng = _Lcg(seed)
    step = span_days / n_nights
    samples = []
    for night in range(n_nights):
        start = round(night * step + rng.next_u() * 1.2, 5)
        for _ in range(per_night):
            t = round(start + rng.next_u() * 0.34, 5)
            th = TWO_PI * (t % period) / period
            m = sum(a * math.cos(k * th) + b * math.sin(k * th) for k, a, b in harmonics)
            m += (rng.next_u() + rng.next_u() - 1.0) * math.sqrt(6.0) * sigma
            samples.append({"t": t, "dmag": round(m, 3), "sigma": sigma})
    for _ in range(n_spikes):
        idx = int(rng.next_u() * len(samples)) % len(samples)
        spike = (3.5 + 6.0 * rng.next_u()) * sigma
        samples[idx]["dmag"] = round(samples[idx]["dmag"] + spike, 3)
    samples.sort(key=lambda s: s["t"])
    return {"site": {"telescope": "b2", "band": "V", "time_system": "HJD"},
            "fit": {"f_min": 0.02, "f_max": 2.0, "n_harmonics": n_harm,
                    "sigma_clip": 3.0, "max_clip_iter": 2, "min_used": 100},
            "samples": samples}


# ---------------------------------------------------------------- 独立复算
def _solve(a, b):
    n = len(a)
    m = [a[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-18:
            return None
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for k in range(col, n + 1):
            m[col][k] /= pv
        for r in range(n):
            if r != col:
                fac = m[r][col]
                for k in range(col, n + 1):
                    m[r][k] -= fac * m[col][k]
    return [m[i][n] for i in range(n)]


def _row(t, f, n_harm):
    th = TWO_PI * f * t
    row = [1.0]
    for k in range(1, n_harm + 1):
        row.append(math.cos(k * th))
        row.append(math.sin(k * th))
    return row


def _fit(points, f, n_harm):
    npar = 2 * n_harm + 1
    xtx = [[0.0] * npar for _ in range(npar)]
    xty = [0.0] * npar
    for t, y, sig in points:
        row = _row(t, f, n_harm)
        w = 1.0 / (sig * sig)
        for i in range(npar):
            xty[i] += w * row[i] * y
            for j in range(npar):
                xtx[i][j] += w * row[i] * row[j]
    coef = _solve(xtx, xty)
    if coef is None:
        return None, None
    chi2 = 0.0
    for t, y, sig in points:
        model = sum(c * r for c, r in zip(coef, _row(t, f, n_harm)))
        chi2 += ((y - model) / sig) ** 2
    return coef, chi2


def _golden(fn, lo, hi, iters=80):
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def expect(data):
    """独立复算：频率步长取得比题面更细（1/(5·跨度)），剔离群点与细化口径相同。"""
    cfg = data["fit"]
    n_harm = int(cfg["n_harmonics"])
    f_min = float(cfg["f_min"])
    f_max = float(cfg["f_max"])
    clip = float(cfg["sigma_clip"])
    max_iter = int(cfg["max_clip_iter"])
    pts = [(float(s["t"]), float(s["dmag"]), float(s["sigma"])) for s in data["samples"]]
    span = max(p[0] for p in pts) - min(p[0] for p in pts)
    df = 1.0 / (5.0 * span)

    best = None
    grid = []
    steps = int(math.floor((f_max - f_min) / df)) + 1
    for i in range(steps + 1):
        f = f_min + i * df
        if f > f_max + 1e-12:
            break
        _, chi2 = _fit(pts, f, n_harm)
        if chi2 is None:
            continue
        grid.append((chi2, f))
        if best is None or chi2 < best[0]:
            best = (chi2, f)
    f_star = best[1]

    keep = list(pts)
    for _ in range(max_iter):
        coef, _ = _fit(keep, f_star, n_harm)
        if coef is None:
            break
        left = []
        dropped = False
        for t, y, sig in keep:
            model = sum(c * r for c, r in zip(coef, _row(t, f_star, n_harm)))
            if abs((y - model) / sig) > clip:
                dropped = True
            else:
                left.append((t, y, sig))
        if not dropped:
            break
        keep = left

    lo = max(f_min, f_star - df)
    hi = min(f_max, f_star + df)
    if hi > lo:
        f_best = _golden(lambda x: _fit(keep, x, n_harm)[1], lo, hi)
    else:
        f_best = f_star
    coef, chi2 = _fit(keep, f_best, n_harm)
    period = 1.0 / f_best
    phi = math.atan2(coef[2], coef[1])
    if phi < 0:
        phi += TWO_PI
    dof = len(keep) - (2 * n_harm + 1)
    h = df / 10.0
    c0 = _fit(keep, f_best, n_harm)[1]
    c_plus = _fit(keep, f_best + h, n_harm)[1]
    c_minus = _fit(keep, f_best - h, n_harm)[1]
    curvature = (c_plus - 2.0 * c0 + c_minus) / (h * h)
    sigma_f = math.sqrt(2.0 / curvature) if curvature > 0 else 0.0
    others = [(c, f) for c, f in grid if abs(f - f_best) > 0.05 * f_best]
    next_period = 1.0 / min(others)[1] if others else period
    return {"period": period, "epoch": (period * phi / TWO_PI) % period,
            "amplitude": math.hypot(coef[1], coef[2]), "chi2": chi2,
            "reduced": chi2 / dof, "n_used": len(keep), "dof": dof,
            "coef": coef, "f": f_best,
            "sigma_period": period * period * sigma_f,
            "next_period": next_period}


# ---------------------------------------------------------------- 驱动被测程序
def _run(path, out_path, cache_path=None, timeout=240):
    cmd = [sys.executable, ENTRY, "--data", path, "--out", out_path]
    if cache_path is not None:
        cmd += ["--cache", cache_path]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def run_case(data, cache_path=None):
    tmp = tempfile.mkdtemp(prefix="lc-case-")
    dpath = os.path.join(tmp, "lcurve.json")
    opath = os.path.join(tmp, "report.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    proc = _run(dpath, opath, cache_path)
    report = None
    if os.path.exists(opath):
        try:
            with open(opath, encoding="utf-8") as f:
                report = json.load(f)
        except ValueError:
            report = None
    return report, {"data": dpath, "out": opath, "proc": proc, "dir": tmp}


NEED = {"n_total", "n_used", "dof", "period_days", "epoch_days",
        "amplitude_mmag", "period_uncertainty_days", "next_candidate_period_days",
        "chi2", "reduced_chi2"}


def _num(report, key):
    assert isinstance(report, dict), "报告不存在或不是 JSON 对象"
    assert key in report, "报告缺少字段 %s" % key
    try:
        v = float(report[key])
    except (TypeError, ValueError):
        raise AssertionError("字段 %s 不是数值：%r" % (key, report[key]))
    assert math.isfinite(v), "字段 %s 不是有限数值：%r" % (key, v)
    return v


# ---------------------------------------------------------------- 测试
def test_period_matches_independent_search():
    """题面原文：「步长 df = 1/(4·观测跨度)……取加权残差平方和最小的频率」。

    出厂曲线（周期 1.8539 天）：报告周期与独立复算相差不超过 0.3%，
    且与真值 1.8539 天相差不超过 1%。
    """
    data = make_curve(seed=101)
    report, _ = run_case(data)
    for key in NEED:
        assert key in report, "报告缺少字段 %s" % key
    exp = expect(data)
    got = _num(report, "period_days")
    assert abs(got - exp["period"]) <= 0.003 * exp["period"], \
        "周期与独立复算差得太多：被测 %r，独立复算 %r" % (got, exp["period"])
    assert abs(got - 1.8539) <= 0.01 * 1.8539, \
        "周期离真值太远：被测 %r，真值 1.8539" % got


def test_half_period_harmonic_rejected():
    """题面原文：「模型是常数项加前 n_harmonics 阶谐波」，脉冲形状不对称时不能按一半周期报。

    报告周期必须落在真周期附近，不能落到一半周期或倍周期上；谐波阶数不够时残差平方和会
    明显偏大，所以 chi2 也要与独立复算相差不超过 10%。
    """
    data = make_curve(seed=211)
    true_p = 1.8539
    exp = expect(data)
    report, _ = run_case(data)
    got = _num(report, "period_days")
    chi2 = _num(report, "chi2")
    assert abs(got - true_p) <= 0.01 * true_p, \
        "周期没有落在真周期上：被测 %r，真值 %r" % (got, true_p)
    assert abs(got - 0.5 * true_p) > 0.05 * true_p, \
        "周期掉到了半周期谐波上：被测 %r" % got
    assert abs(got - 2.0 * true_p) > 0.05 * true_p, \
        "周期掉到了倍周期上：被测 %r" % got
    assert abs(chi2 - exp["chi2"]) <= 0.10 * exp["chi2"], \
        "chi2 与独立复算差得太多（谐波阶数是不是不够）：被测 %r，独立复算 %r" % (chi2, exp["chi2"])


def test_epoch_and_amplitude_match():
    """题面原文：「epoch_days 落在 [0, P)」「amplitude_mmag = sqrt(a1² + b1²)」。

    极大时刻与独立复算相差不超过 3%（按周期折算），半振幅相差不超过 6%。
    """
    data = make_curve(seed=311)
    exp = expect(data)
    report, _ = run_case(data)
    period = _num(report, "period_days")
    epoch = _num(report, "epoch_days")
    amp = _num(report, "amplitude_mmag")
    assert 0.0 <= epoch < period, "epoch_days 应落在 [0, P)：%r 不在 [0, %r)" % (epoch, period)
    diff = abs(epoch - exp["epoch"])
    diff = min(diff, period - diff)
    assert diff <= 0.03 * period, \
        "极大时刻偏了：被测 %r，独立复算 %r" % (epoch, exp["epoch"])
    assert abs(amp - exp["amplitude"]) <= 0.06 * exp["amplitude"], \
        "半振幅偏了：被测 %r，独立复算 %r" % (amp, exp["amplitude"])


def test_outliers_clipped_before_fit():
    """题面原文：「绝对值超过 sigma_clip 的整点剔除，然后重拟合」」。

    曲线里有 9 个宇宙线尖峰：n_used 必须比 n_total 少掉这些点（独立复算差一个点以内），
    且 chi2 落回噪声量级（reduced_chi2 不超过 1.6）。
    """
    data = make_curve(seed=409)
    exp = expect(data)
    report, _ = run_case(data)
    n_total = int(_num(report, "n_total"))
    n_used = int(_num(report, "n_used"))
    assert n_total == len(data["samples"]), \
        "n_total 应为观测点总数 %d，实际 %d" % (len(data["samples"]), n_total)
    assert abs(n_used - exp["n_used"]) <= 1, \
        "剔除后的有效点数不对：被测 %r，独立复算 %r" % (n_used, exp["n_used"])
    assert n_used < n_total, "一个离群点都没剔：n_used=%r n_total=%r" % (n_used, n_total)
    red = _num(report, "reduced_chi2")
    assert 0.4 <= red <= 1.6, "reduced_chi2 不在噪声量级：%r" % red


def test_second_star_other_period():
    """题面原文：「复检会喂别的星的曲线，周期、脉冲形状、观测跨度、噪声水平与离群点都不同」。

    换一颗周期 0.6213 天、两阶谐波、更短跨度、更小噪声的星，周期与独立复算相差不超过 0.3%。
    """
    data = make_curve(period=0.6213, span_days=60.0, n_nights=30, per_night=12,
                      sigma=12.0, harmonics=((1, 30.0, -18.0), (2, 9.0, 4.0)),
                      n_spikes=6, n_harm=2, seed=503)
    report, _ = run_case(data)
    exp = expect(data)
    got = _num(report, "period_days")
    assert abs(got - exp["period"]) <= 0.003 * exp["period"], \
        "换星后周期偏了：被测 %r，独立复算 %r" % (got, exp["period"])
    n_used = int(_num(report, "n_used"))
    assert abs(n_used - exp["n_used"]) <= 1, \
        "换星后有效点数不对：被测 %r，独立复算 %r" % (n_used, exp["n_used"])


def test_long_period_near_low_frequency_edge():
    """题面原文：「从 f_min 起逐步加大，扫到 f_max 为止」——长周期星的频率贴着下边界。

    周期 48 天（频率 0.0208，只比 f_min=0.02 高半个粗步长）：频率搜索不能跳过边界点，
    周期要落在独立复算 0.5% 以内。
    """
    data = make_curve(period=48.0, span_days=120.0, n_nights=40, per_night=10,
                      sigma=10.0, harmonics=((1, 25.0, 12.0), (2, -6.0, 3.0)),
                      n_spikes=0, n_harm=2, seed=607)
    report, _ = run_case(data)
    exp = expect(data)
    got = _num(report, "period_days")
    assert abs(got - exp["period"]) <= 0.005 * exp["period"], \
        "贴边界的周期偏了：被测 %r，独立复算 %r" % (got, exp["period"])
    assert got > 1.0 / data["fit"]["f_max"], "周期越过了频率上边界"


def test_chi2_and_dof_contract():
    """题面原文：「dof = n_used - (2·H + 1)」「reduced_chi2 = chi2 / dof」。

    dof 与 n_used、谐波阶数一致，reduced_chi2 与 chi2/dof 一致（相对 0.1%），
    且 chi2 与独立复算相差不超过 5%。
    """
    data = make_curve(seed=701)
    exp = expect(data)
    report, _ = run_case(data)
    n_used = int(_num(report, "n_used"))
    dof = int(_num(report, "dof"))
    chi2 = _num(report, "chi2")
    red = _num(report, "reduced_chi2")
    assert dof == n_used - 7, "dof 应等于 n_used - 7：n_used=%r dof=%r" % (n_used, dof)
    assert abs(red - chi2 / dof) <= 0.001 * (chi2 / dof), \
        "reduced_chi2 与 chi2/dof 不一致：%r 对 %r" % (red, chi2 / dof)
    assert abs(chi2 - exp["chi2"]) <= 0.05 * exp["chi2"], \
        "chi2 与独立复算差得太多：被测 %r，独立复算 %r" % (chi2, exp["chi2"])


def test_period_uncertainty_from_curvature():
    """题面原文：「sigma_f = sqrt(2/c)、sigma_P = P²·sigma_f」。

    曲率法给出的周期不确定度必须为正、与独立复算相差不超过 15%，且比周期小三四个量级。
    """
    data = make_curve(seed=1801)
    exp = expect(data)
    report, _ = run_case(data)
    got = _num(report, "period_uncertainty_days")
    period = _num(report, "period_days")
    assert got > 0.0, "周期不确定度必须为正：%r" % got
    assert got < 0.05 * period, "周期不确定度比周期本身还大：%r" % got
    assert abs(got - exp["sigma_period"]) <= 0.15 * exp["sigma_period"], \
        "周期不确定度与独立复算差得太多：被测 %r，独立复算 %r" % (got, exp["sigma_period"])


def test_next_candidate_is_reported():
    """题面原文：「取剩下的点里残差平方和最小的那个频率……用来判断这次定出的周期有没有被谐波或日周期缺测的假峰抢走」。

    次优候选不能就是最优周期本身，而且要落在最优周期的整数倍或整数分之一上（谐波族），
    说明程序确实在粗扫网格上找到了那条与主峰竞争的假峰。
    """
    data = make_curve(seed=1901)
    report, _ = run_case(data)
    got = _num(report, "next_candidate_period_days")
    best = _num(report, "period_days")
    assert got > 0.0, "次优候选周期必须为正：%r" % got
    assert abs(got - best) > 0.01 * best, "次优候选就是最优周期本身：%r" % got
    ratios = (0.25, 1.0 / 3.0, 0.5, 1.5, 2.0, 3.0, 4.0)
    ratio = got / best
    hit = [r for r in ratios if abs(ratio - r) <= 0.05 * r]
    assert hit, "次优候选周期不在谐波族上：候选 %r / 最优 %r = %r" % (got, best, ratio)


def test_invalid_data_rejected_without_output():
    """题面原文：「记录不合法时以退出码 2 结束、往 stderr 写原因、不产出报告」。

    五种坏记录：观测点不足 100、时间不单调、sigma 非正、f_min 不小于 f_max、有效点不足。
    每种都要退出码 2、stderr 有原因、目录里不留报告也不留临时文件。
    """
    good = make_curve(seed=809, n_spikes=0)
    cases = []
    few = json.loads(json.dumps(good))
    few["samples"] = few["samples"][:60]
    cases.append(("观测点不足", few))
    nonmono = json.loads(json.dumps(good))
    nonmono["samples"][10]["t"] = nonmono["samples"][9]["t"]
    cases.append(("时间不单调", nonmono))
    badsig = json.loads(json.dumps(good))
    badsig["samples"][5]["sigma"] = 0.0
    cases.append(("sigma 非正", badsig))
    badrange = json.loads(json.dumps(good))
    badrange["fit"]["f_min"] = 3.0
    cases.append(("频率范围颠倒", badrange))
    short = json.loads(json.dumps(good))
    short["fit"]["min_used"] = 2000
    cases.append(("有效点不足", short))
    for label, data in cases:
        tmp = tempfile.mkdtemp(prefix="lc-bad-")
        dpath = os.path.join(tmp, "lcurve.json")
        opath = os.path.join(tmp, "report.json")
        with open(dpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        proc = _run(dpath, opath)
        assert proc.returncode == 2, "%s：退出码应为 2，实际 %r" % (label, proc.returncode)
        assert (proc.stderr or "").strip(), "%s：stderr 没有写原因" % label
        assert not os.path.exists(opath), "%s：不该产出报告文件" % label
        leftovers = [n for n in os.listdir(tmp) if n != "lcurve.json"]
        assert not leftovers, "%s：目录里留下残留文件 %r" % (label, leftovers)


def test_report_schema_and_rounding():
    """题面原文：「period_days 保留五位小数；epoch_days 保留四位；amplitude_mmag、chi2 保留两位；reduced_chi2 保留四位」。

    校验顶层字段、位数与有限性，meta.model 写成 harmonics-<H>，文件里不出现 NaN 或 Infinity。
    """
    data = make_curve(seed=907)
    report, ctx = run_case(data)
    for key in NEED:
        assert key in report, "报告缺少字段 %s" % key
    for key, digits in (("period_days", 5), ("period_uncertainty_days", 5),
                        ("next_candidate_period_days", 5), ("epoch_days", 4),
                        ("amplitude_mmag", 2), ("chi2", 2), ("reduced_chi2", 4)):
        v = _num(report, key)
        assert abs(v - round(v, digits)) <= 1e-12, "字段 %s 不是 %d 位小数：%r" % (key, digits, v)
    meta = report.get("meta")
    assert isinstance(meta, dict), "meta 应是对象"
    assert meta.get("model") == "harmonics-3", \
        "meta.model 应写成 harmonics-3，实际 %r" % (meta.get("model"),)
    raw = open(ctx["out"], encoding="utf-8").read()
    assert "NaN" not in raw and "Infinity" not in raw, "报告里出现了 NaN 或 Infinity"


def test_rerun_produces_identical_bytes():
    """题面原文：「同一份记录重复运行，报告逐字节一致（不要写时间戳之类的运行期信息）」。

    连跑两次，两个报告文件字节完全相同。
    """
    data = make_curve(seed=1009)
    first, ctx1 = run_case(data)
    for key in NEED:
        assert key in first, "第一次运行就缺字段 %s" % key
    body1 = open(ctx1["out"], "rb").read()
    second, ctx2 = run_case(data)
    assert second is not None, "第二次运行没有产出报告"
    body2 = open(ctx2["out"], "rb").read()
    assert body1 == body2, "两次运行报告字节不一致（是不是写了时间戳）"


def test_failed_run_keeps_previous_report():
    """题面原文：「运行失败时不能覆盖上一次已经写好的报告」。

    先跑一次合格记录留下报告，再用时间不单调的坏记录跑：退出码 2，旧报告字节不变。
    """
    good = make_curve(seed=1103, n_spikes=0)
    tmp = tempfile.mkdtemp(prefix="lc-keep-")
    dpath = os.path.join(tmp, "lcurve.json")
    opath = os.path.join(tmp, "report.json")
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False)
    first = _run(dpath, opath)
    assert first.returncode == 0, "合格记录应退出 0，实际 %r" % first.returncode
    assert os.path.exists(opath), "合格记录没有产出报告"
    before = open(opath, "rb").read()
    broken = json.loads(json.dumps(good))
    broken["samples"][5]["t"] = broken["samples"][4]["t"]
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    second = _run(dpath, opath)
    assert second.returncode == 2, "坏记录应退出 2，实际 %r" % second.returncode
    assert open(opath, "rb").read() == before, "运行失败把上一次的报告覆盖了"


def test_cache_reuse_and_append():
    """题面原文：「这个 key 已在缓存里就原地替换该条目的 report；不在就追加到末尾」。

    连跑三次：同一份记录跑两次（中间把缓存里那条报告的 period_days 改成哨兵值 99.0），
    第二次必须直接复用缓存、报出哨兵值；换一份记录再跑，缓存变成两条、顺序不变。
    """
    import hashlib
    cache = os.path.join(tempfile.mkdtemp(prefix="lc-cache-"), "cache.json")
    data_a = make_curve(seed=1301)
    report_a, ctx_a = run_case(data_a, cache_path=cache)
    for key in NEED:
        assert key in report_a, "第一次运行就缺字段 %s" % key
    assert os.path.exists(cache), "运行结束没有写缓存文件"
    with open(cache, encoding="utf-8") as f:
        body = json.load(f)
    assert isinstance(body, dict) and body.get("version") == 1, "缓存结构不对：%r" % (list(body)[:4],)
    entries = body.get("entries")
    assert isinstance(entries, list) and len(entries) == 1, "缓存条目数应为 1，实际 %r" % (entries,)
    key_a = hashlib.sha256(open(ctx_a["data"], "rb").read()).hexdigest()
    assert entries[0].get("key") == key_a, "缓存 key 不是记录字节的 sha256：%r" % (entries[0].get("key"),)
    assert isinstance(entries[0].get("report"), dict), "缓存条目里的 report 不是对象"

    entries[0]["report"]["period_days"] = 99.0
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False)
    report_again, _ = run_case(data_a, cache_path=cache)
    assert float(report_again.get("period_days", 0.0)) == 99.0, \
        "第二次跑同一份记录没有复用缓存：period_days=%r" % (report_again.get("period_days"),)

    data_b = make_curve(period=0.6213, span_days=60.0, n_nights=30, per_night=12,
                        sigma=12.0, harmonics=((1, 30.0, -18.0), (2, 9.0, 4.0)),
                        n_spikes=6, n_harm=2, seed=1409)
    report_b, ctx_b = run_case(data_b, cache_path=cache)
    exp_b = expect(data_b)
    assert abs(float(report_b["period_days"]) - exp_b["period"]) <= 0.01 * exp_b["period"], \
        "换记录后没有重新搜索：period_days=%r，独立复算 %r" % (report_b.get("period_days"), exp_b["period"])
    with open(cache, encoding="utf-8") as f:
        body2 = json.load(f)
    keys = [e.get("key") for e in body2.get("entries", [])]
    assert len(keys) == 2, "缓存条目数应为 2，实际 %r" % (keys,)
    assert keys[0] == key_a, "原有条目被挪动或覆盖了：%r" % (keys,)
    key_b = hashlib.sha256(open(ctx_b["data"], "rb").read()).hexdigest()
    assert keys[1] == key_b, "新条目没有追加到末尾：%r" % (keys,)


def test_corrupt_cache_recovered():
    """题面原文：「缓存文件读不出来、不是合法 JSON 或结构不对：当作空缓存」。

    先塞一段乱码当缓存，再跑一份合格记录：退出码 0、报告正确、缓存被重写成合法结构。
    """
    cache = os.path.join(tempfile.mkdtemp(prefix="lc-cache-bad-"), "cache.json")
    with open(cache, "w", encoding="utf-8") as f:
        f.write("{ this is not json ][")
    data = make_curve(seed=1511)
    report, _ = run_case(data, cache_path=cache)
    exp = expect(data)
    assert isinstance(report, dict), "坏缓存把整次运行拖挂了（没有报告）"
    assert abs(float(report["period_days"]) - exp["period"]) <= 0.01 * exp["period"], \
        "坏缓存下报告不对：period_days=%r，独立复算 %r" % (report.get("period_days"), exp["period"])
    with open(cache, encoding="utf-8") as f:
        body = json.load(f)
    assert body.get("version") == 1, "坏缓存没有被重写成合法结构：%r" % (body,)
    assert len(body.get("entries", [])) == 1, "重写后的缓存应只有一条，实际 %r" % (body.get("entries"),)


def test_cache_capped_at_32_entries():
    """题面原文：「缓存最多留 32 条：追加新条目时如果会超过 32 条，从最早的那头丢掉」。

    先往缓存里塞满 32 条旧条目，再跑一份新记录：缓存仍是 32 条 —— 最早那条被丢掉、
    新的一条在末尾、原来第 2 到第 32 条顺序不变。
    """
    import hashlib
    cache = os.path.join(tempfile.mkdtemp(prefix="lc-cache-cap-"), "cache.json")
    stub = {"n_total": 400, "n_used": 391, "dof": 384, "period_days": 1.85315,
            "epoch_days": 0.0832, "amplitude_mmag": 42.01,
            "period_uncertainty_days": 0.00044, "next_candidate_period_days": 3.70258,
            "chi2": 411.8, "reduced_chi2": 1.0724,
            "meta": {"model": "harmonics-3", "unit": "HJD"}}
    old_keys = ["%064x" % (i + 1) for i in range(32)]
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"version": 1,
                   "entries": [{"key": k, "report": stub} for k in old_keys]}, f)

    data = make_curve(seed=3000, n_spikes=0)
    report, ctx = run_case(data, cache_path=cache)
    assert isinstance(report, dict), "新记录没有产出报告"
    with open(cache, encoding="utf-8") as f:
        body = json.load(f)
    entries = body.get("entries")
    assert isinstance(entries, list), "缓存结构不对：%r" % (body,)
    assert len(entries) == 32, "缓存应只留 32 条，实际 %d 条" % len(entries)
    got = [e.get("key") for e in entries]
    new_key = hashlib.sha256(open(ctx["data"], "rb").read()).hexdigest()
    assert got[-1] == new_key, "新条目没有追加到末尾：%r" % (got[-1],)
    assert old_keys[0] not in got, "最早的一条没有被丢掉"
    assert got[:-1] == old_keys[1:], "留下的不是原来第 2 到第 32 条（顺序也不对）"


def test_failed_run_leaves_cache_untouched():
    """题面原文：「记录不合法（退出码 2）时缓存文件一个字节都不许动」。

    先跑一份合格记录把缓存建起来，再用坏记录跑：退出码 2，缓存字节与旧报告都不变；
    再把缓存路径指成一个目录（写不进去），也要按非法记录退出 2、不产出报告。
    """
    cache = os.path.join(tempfile.mkdtemp(prefix="lc-cache-keep-"), "cache.json")
    good = make_curve(seed=1601, n_spikes=0)
    report, ctx = run_case(good, cache_path=cache)
    assert isinstance(report, dict), "合格记录没有产出报告"
    before_cache = open(cache, "rb").read()
    before_report = open(ctx["out"], "rb").read()

    broken = json.loads(json.dumps(good))
    broken["samples"][7]["t"] = broken["samples"][6]["t"]
    bad_dir = tempfile.mkdtemp(prefix="lc-cache-broken-")
    bad_data = os.path.join(bad_dir, "lcurve.json")
    bad_out = os.path.join(bad_dir, "report.json")
    with open(bad_data, "w", encoding="utf-8") as f:
        json.dump(broken, f, ensure_ascii=False)
    proc = _run(bad_data, bad_out, cache_path=cache)
    assert proc.returncode == 2, "坏记录应退出 2，实际 %r" % proc.returncode
    assert open(cache, "rb").read() == before_cache, "非法记录把缓存改动了"
    assert open(ctx["out"], "rb").read() == before_report, "非法记录把旧报告改动了"
    assert not os.path.exists(bad_out), "非法记录不该产出报告"

    unwritable = tempfile.mkdtemp(prefix="lc-cache-dir-")
    good2 = make_curve(seed=1709, n_spikes=0)
    data2 = os.path.join(unwritable, "lcurve.json")
    out2 = os.path.join(unwritable, "report.json")
    with open(data2, "w", encoding="utf-8") as f:
        json.dump(good2, f, ensure_ascii=False)
    proc2 = _run(data2, out2, cache_path=unwritable)
    assert proc2.returncode == 2, "缓存写不进去时应退出 2，实际 %r" % proc2.returncode
    assert not os.path.exists(out2), "缓存写不进去时不该产出报告"
