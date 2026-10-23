#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变星光变曲线周期复核：三阶谐波加权拟合 + 周期搜索 + 夜间账本。

命令行：
    python3 src/period.py --data data/lcurve.json --out out/report.json --ledger out/ledger.json

口径：
  · 相位 th = 2π·(t mod P)/P，模型 m(t) = c0 + Σ_{k=1..H} [a_k·cos(k·th) + b_k·sin(k·th)]，
    H 取记录里的 fit.n_harmonics（本批是 3）；
  · 权重按每个点的 sigma 倒数平方；给定周期时七个系数是加权线性最小二乘（闭式解）；
  · 周期搜索：先用 df = 1/(4·观测跨度) 的频率步长在 [f_min, f_max] 上扫一遍，
    找到残差平方和最小的频率；再在最优频率上按 sigma_clip 反复剔掉离群点
    （最多 max_clip_iter 轮）；最后在最优频率 ±df 内用黄金分割细化；
  · epoch_days 是基波极大时刻（落在 [0, P)），amplitude_mmag 是基波半振幅；
  · dof = n_used - 7，reduced_chi2 = chi2 / dof；
  · 夜间账本按夜号（floor(t)）汇总本次记录：点数、保留点数、保留点标准化残差绝对值中位数，
    与旧账本按夜号合并（同夜号整条替换、按夜号升序、最多留 64 夜）；
  · 输出先写临时文件再原子替换；非法输入以退出码 2 结束且不落文件。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

EXIT_INVALID = 2
LEDGER_LIMIT = 64
LEDGER_FORMAT = "night-ledger"
TWO_PI = 2.0 * math.pi


def _fail(msg):
    sys.stderr.write("非法输入：%s\n" % msg)
    raise SystemExit(EXIT_INVALID)


def load_data(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        _fail("读不到记录 %s" % path)
    except ValueError as e:
        _fail("记录不是合法 JSON：%s" % e)
    if not isinstance(raw, dict):
        _fail("记录顶层要是对象")
    cfg = raw.get("fit")
    if not isinstance(cfg, dict):
        _fail("缺 fit 配置")
    f_min = cfg.get("f_min")
    f_max = cfg.get("f_max")
    n_harm = cfg.get("n_harmonics", 3)
    clip = cfg.get("sigma_clip", 3.0)
    max_iter = cfg.get("max_clip_iter", 2)
    min_used = cfg.get("min_used", 100)
    for name, val in (("f_min", f_min), ("f_max", f_max)):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(float(val)):
            _fail("fit.%s 要是有限数值" % name)
    f_min = float(f_min)
    f_max = float(f_max)
    if not (f_min > 0 and f_max > f_min):
        _fail("fit 要满足 0 < f_min < f_max")
    if not isinstance(n_harm, int) or isinstance(n_harm, bool) or n_harm < 1:
        _fail("fit.n_harmonics 要是不小于 1 的整数")
    if not isinstance(clip, (int, float)) or isinstance(clip, bool) or not (float(clip) > 0):
        _fail("fit.sigma_clip 要是正数")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 0:
        _fail("fit.max_clip_iter 要是不小于 0 的整数")
    if not isinstance(min_used, int) or isinstance(min_used, bool) or min_used < 20:
        _fail("fit.min_used 要是不小于 20 的整数")

    samples = raw.get("samples")
    if not isinstance(samples, list) or len(samples) < 100:
        _fail("samples 至少要 100 个观测点")
    times, mags, sigmas = [], [], []
    for i, s in enumerate(samples):
        if not isinstance(s, dict):
            _fail("samples[%d] 不是对象" % i)
        try:
            t = float(s["t"])
            y = float(s["dmag"])
            sig = float(s["sigma"])
        except (KeyError, TypeError, ValueError):
            _fail("samples[%d] 的 t/dmag/sigma 必须是数值" % i)
        if not (math.isfinite(t) and math.isfinite(y) and math.isfinite(sig)):
            _fail("samples[%d] 有非有限数值" % i)
        if sig <= 0:
            _fail("samples[%d].sigma 必须为正" % i)
        if t < 0 or (times and t <= times[-1]):
            _fail("samples[%d].t 必须单调递增且非负" % i)
        times.append(t)
        mags.append(y)
        sigmas.append(sig)
    return {"times": times, "mags": mags, "sigmas": sigmas,
            "nights": [int(math.floor(t)) for t in times],
            "f_min": f_min, "f_max": f_max, "n_harm": int(n_harm),
            "clip": float(clip), "max_iter": int(max_iter),
            "min_used": int(min_used), "site": raw.get("site") or {}}


def _design_row(t, f, n_harm):
    th = TWO_PI * f * t
    row = [1.0]
    for k in range(1, n_harm + 1):
        row.append(math.cos(k * th))
        row.append(math.sin(k * th))
    return row


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


def fit_at(ts, ys, sigmas, nights, f, n_harm):
    """给定频率做加权谐波最小二乘，夜零点偏移一并解出来。

    做法是「去夜均值」：每夜内把设计列与观测各自减去该夜的加权均值，常数项与零点偏移
    一起被差掉，剩下的正规方程只解 2H 个谐波系数；再回填 c0（加权总均值）与每夜偏移
    （该夜残差的加权均值，按构造对全部观测点的加权和为零）。
    返回 (coef, c0, offsets, chi2)。
    """
    npar = 2 * n_harm + 1
    per = {}
    for t, y, sig, night in zip(ts, ys, sigmas, nights):
        row = _design_row(t, f, n_harm)
        w = 1.0 / (sig * sig)
        slot = per.get(night)
        if slot is None:
            slot = per[night] = {"sw": 0.0, "cols": [0.0] * npar, "sy": 0.0}
        slot["sw"] += w
        slot["sy"] += w * y
        for i in range(npar):
            slot["cols"][i] += w * row[i]

    m = npar - 1
    xtx = [[0.0] * m for _ in range(m)]
    xty = [0.0] * m
    rows = []
    for t, y, sig, night in zip(ts, ys, sigmas, nights):
        slot = per[night]
        mean_cols = [v / slot["sw"] for v in slot["cols"]]
        row = _design_row(t, f, n_harm)
        rows.append((row, y, 1.0 / (sig * sig), night, mean_cols))
    for row, y, w, night, mean_cols in rows:
        x = [row[i + 1] - mean_cols[i + 1] for i in range(m)]
        dy = y - per[night]["sy"] / per[night]["sw"]
        for i in range(m):
            xty[i] += w * x[i] * dy
            for j in range(m):
                xtx[i][j] += w * x[i] * x[j]
    beta = _solve(xtx, xty)
    if beta is None:
        return None, None, None, None
    coef = [0.0] + beta

    sw_all = 0.0
    acc = 0.0
    residual = []
    for row, y, w, night, mean_cols in rows:
        model = sum(c * r for c, r in zip(coef, row))
        residual.append(y - model)
        sw_all += w
        acc += w * (y - model)
    c0 = acc / sw_all if sw_all else 0.0

    night_sum = {}
    for (row, y, w, night, mean_cols), r in zip(rows, residual):
        slot = night_sum.get(night)
        if slot is None:
            slot = night_sum[night] = [0.0, 0.0]
        slot[0] += w * (r - c0)
        slot[1] += w
    offsets = {night: v[0] / v[1] for night, v in night_sum.items()}

    chi2 = 0.0
    for (row, y, w, night, mean_cols), r in zip(rows, residual):
        d = r - c0 - offsets[night]
        chi2 += w * d * d
    return coef, c0, offsets, chi2


def _chi2_only(ts, ys, sigmas, nights, f, n_harm):
    _, _, _, chi2 = fit_at(ts, ys, sigmas, nights, f, n_harm)
    if chi2 is None:
        return float("inf")
    return chi2


def _golden(fn, lo, hi, iters=90):
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


def search_period(scan):
    ts = scan["times"]
    ys = scan["mags"]
    sigmas = scan["sigmas"]
    nights = scan["nights"]
    n_harm = scan["n_harm"]
    span = max(ts) - min(ts)
    df = 1.0 / (4.0 * span)

    best = None
    grid = []
    steps = int(math.floor((scan["f_max"] - scan["f_min"]) / df)) + 1
    for i in range(steps + 1):
        f = scan["f_min"] + i * df
        if f > scan["f_max"] + 1e-12:
            break
        _, _, _, chi2 = fit_at(ts, ys, sigmas, nights, f, n_harm)
        if chi2 is None:
            continue
        grid.append((chi2, f))
        if best is None or chi2 < best[0]:
            best = (chi2, f)
    if best is None:
        _fail("周期搜索没找到可行解，检查 f_min / f_max")
    f_star = best[1]

    keep = list(range(len(ts)))
    for _ in range(scan["max_iter"]):
        kt = [ts[i] for i in keep]
        ky = [ys[i] for i in keep]
        ks = [sigmas[i] for i in keep]
        kn = [nights[i] for i in keep]
        coef, c0, offsets, _ = fit_at(kt, ky, ks, kn, f_star, n_harm)
        if coef is None:
            break
        drop = []
        for i in keep:
            row = _design_row(ts[i], f_star, n_harm)
            model = sum(c * r for c, r in zip(coef, row))
            if abs((ys[i] - c0 - offsets[nights[i]] - model) / sigmas[i]) > scan["clip"]:
                drop.append(i)
        if not drop:
            break
        keep = [i for i in keep if i not in set(drop)]

    if len(keep) < scan["min_used"]:
        _fail("剔除离群点后只剩 %d 个有效点，少于要求的 %d" % (len(keep), scan["min_used"]))

    kt = [ts[i] for i in keep]
    ky = [ys[i] for i in keep]
    ks = [sigmas[i] for i in keep]
    kn = [nights[i] for i in keep]
    lo = max(scan["f_min"], f_star - df)
    hi = min(scan["f_max"], f_star + df)
    if hi <= lo:
        f_best = f_star
    else:
        f_best = _golden(lambda x: _chi2_only(kt, ky, ks, kn, x, n_harm), lo, hi)
    coef, c0, offsets, chi2 = fit_at(kt, ky, ks, kn, f_best, n_harm)
    if coef is None:
        _fail("最终拟合解不出来，检查记录")
    h = df / 10.0
    c0_chi = _chi2_only(kt, ky, ks, kn, f_best, n_harm)
    c_plus = _chi2_only(kt, ky, ks, kn, f_best + h, n_harm)
    c_minus = _chi2_only(kt, ky, ks, kn, f_best - h, n_harm)
    curvature = (c_plus - 2.0 * c0_chi + c_minus) / (h * h)
    if curvature > 0:
        sigma_f = math.sqrt(2.0 / curvature)
    else:
        sigma_f = 0.0
    period = 1.0 / f_best
    others = [(c, f) for c, f in grid if abs(f - f_best) > 0.05 * f_best]
    next_period = 1.0 / min(others)[1] if others else period
    return {"f": f_best, "period": period, "coef": coef, "chi2": chi2, "c0": c0,
            "offsets": offsets, "n_used": len(keep), "n_total": len(ts), "n_harm": n_harm,
            "keep": keep, "nights_used": sorted(set(kn)),
            "sigma_period": period * period * sigma_f, "next_period": next_period}


def build_report(scan, res):
    n_used = res["n_used"]
    n_nights = len(res["nights_used"])
    dof = n_used - (2 * res["n_harm"] + n_nights)
    if dof <= 0:
        _fail("有效点太少，自由度不够")
    a1 = res["coef"][1]
    b1 = res["coef"][2]
    phi = math.atan2(b1, a1)
    if phi < 0:
        phi += TWO_PI
    epoch = (res["period"] * phi / TWO_PI) % res["period"]
    return {
        "n_total": res["n_total"],
        "n_used": n_used,
        "dof": dof,
        "period_days": round(res["period"], 5),
        "epoch_days": round(epoch, 4),
        "amplitude_mmag": round(math.hypot(a1, b1), 2),
        "period_uncertainty_days": round(res["sigma_period"], 5),
        "next_candidate_period_days": round(res["next_period"], 5),
        "night_offsets": [{"night": night, "offset_mmag": round(res["offsets"][night], 3)}
                          for night in res["nights_used"]],
        "chi2": round(res["chi2"], 2),
        "reduced_chi2": round(res["chi2"] / dof, 4),
        "meta": {"model": "harmonics-%d" % res["n_harm"], "unit": scan["site"].get("time_system", "HJD")},
    }


def night_stats(scan, res):
    """按夜号汇总本次记录：点数、保留点数、该夜的零点偏移估计。"""
    points = {}
    for night in scan["nights"]:
        points[night] = points.get(night, 0) + 1
    kept = {}
    for i in res["keep"]:
        night = scan["nights"][i]
        kept[night] = kept.get(night, 0) + 1
    out = []
    for night in sorted(points):
        out.append({"night": night,
                    "points": points[night],
                    "kept": kept.get(night, 0),
                    "offset_mmag": round(res["offsets"].get(night, 0.0), 3)})
    return out


def load_ledger(path):
    """读夜间账本；读不出来、不是合法 JSON 或结构不合约定都当空账本。"""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []
    if not (isinstance(raw, dict) and raw.get("format") == LEDGER_FORMAT):
        return []
    nights = raw.get("nights")
    if not isinstance(nights, list):
        return []
    out = []
    for e in nights:
        if not isinstance(e, dict):
            continue
        night = e.get("night")
        points = e.get("points")
        kept = e.get("kept")
        off = e.get("offset_mmag")
        if not isinstance(night, int) or isinstance(night, bool):
            continue
        if not isinstance(points, int) or isinstance(points, bool):
            continue
        if not isinstance(kept, int) or isinstance(kept, bool):
            continue
        if not isinstance(off, (int, float)) or isinstance(off, bool):
            continue
        out.append({"night": night, "points": points, "kept": kept,
                    "offset_mmag": round(float(off), 3)})
    return out


def merge_ledger(old, new):
    """同夜号整条替换，按夜号升序排，最多留 LEDGER_LIMIT 夜（超出先丢夜号最小的）。"""
    by_night = {}
    for entry in old:
        by_night[entry["night"]] = entry
    for entry in new:
        by_night[entry["night"]] = entry
    merged = [by_night[night] for night in sorted(by_night)]
    if len(merged) > LEDGER_LIMIT:
        merged = merged[-LEDGER_LIMIT:]
    return merged


def save_ledger(path, entries):
    atomic_write(path, {"format": LEDGER_FORMAT, "nights": entries})


def atomic_write(path, obj):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    body = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".lc-", suffix=".tmp", dir=out_dir or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="光变曲线周期搜索")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default=None)
    args = ap.parse_args(argv)

    scan = load_data(args.data)
    res = search_period(scan)
    report = build_report(scan, res)

    if args.ledger:
        entries = merge_ledger(load_ledger(args.ledger), night_stats(scan, res))
        try:
            save_ledger(args.ledger, entries)
        except OSError:
            _fail("账本写不进去：%s" % args.ledger)

    try:
        atomic_write(args.out, report)
    except OSError:
        _fail("报告写不进去：%s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
