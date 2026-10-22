#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变星光变曲线周期复核：三阶谐波加权拟合 + 周期搜索。

命令行：
    python3 src/period.py --data data/lcurve.json --out out/report.json

口径：
  · 相位 th = 2π·(t mod P)/P，模型 m(t) = c0 + Σ_{k=1..H} [a_k·cos(k·th) + b_k·sin(k·th)]，
    H 取记录里的 fit.n_harmonics（本批是 3）；
  · 权重按每个点的 sigma 倒数平方；给定周期时七个系数是加权线性最小二乘（闭式解）；
  · 周期搜索：先用 df = 1/(4·观测跨度) 的频率步长在 [f_min, f_max] 上扫一遍，
    找到残差平方和最小的频率；再在最优频率上按 sigma_clip 反复剔掉离群点
    （最多 max_clip_iter 轮）；最后在最优频率 ±df 内用黄金分割细化；
  · epoch_days 是基波极大时刻（落在 [0, P)），amplitude_mmag 是基波半振幅；
  · dof = n_used - 7，reduced_chi2 = chi2 / dof；
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
    if f_min <= 0 or f_max <= f_min:
        _fail("fit 要满足 0 < f_min < f_max")
    if not isinstance(n_harm, int) or isinstance(n_harm, bool) or n_harm < 1:
        _fail("fit.n_harmonics 要是不小于 1 的整数")
    if not isinstance(clip, (int, float)) or isinstance(clip, bool) or float(clip) <= 0:
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


def fit_at(ts, ys, sigmas, f, n_harm):
    """给定频率做加权谐波最小二乘，返回系数与 chi2。"""
    npar = 2 * n_harm + 1
    xtx = [[0.0] * npar for _ in range(npar)]
    xty = [0.0] * npar
    rows = []
    for t, y, sig in zip(ts, ys, sigmas):
        row = _design_row(t, f, n_harm)
        rows.append(row)
        w = 1.0 / (sig * sig)
        for i in range(npar):
            xty[i] += w * row[i] * y
            for j in range(npar):
                xtx[i][j] += w * row[i] * row[j]
    coef = _solve(xtx, xty)
    if coef is None:
        return None, None
    chi2 = 0.0
    for row, y, sig in zip(rows, ys, sigmas):
        model = sum(c * r for c, r in zip(coef, row))
        chi2 += ((y - model) / sig) ** 2
    return coef, chi2


def _chi2_only(ts, ys, sigmas, f, n_harm):
    coef, chi2 = fit_at(ts, ys, sigmas, f, n_harm)
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
    n_harm = scan["n_harm"]
    span = max(ts) - min(ts)
    df = 1.0 / (4.0 * span)

    best = None
    steps = int(math.floor((scan["f_max"] - scan["f_min"]) / df)) + 1
    for i in range(steps + 1):
        f = scan["f_min"] + i * df
        if f > scan["f_max"] + 1e-12:
            break
        _, chi2 = fit_at(ts, ys, sigmas, f, n_harm)
        if chi2 is not None and (best is None or chi2 < best[0]):
            best = (chi2, f)
    if best is None:
        _fail("周期搜索没找到可行解，检查 f_min / f_max")
    f_star = best[1]

    keep = list(range(len(ts)))
    for _ in range(scan["max_iter"]):
        kt = [ts[i] for i in keep]
        ky = [ys[i] for i in keep]
        ks = [sigmas[i] for i in keep]
        coef, _ = fit_at(kt, ky, ks, f_star, n_harm)
        if coef is None:
            break
        drop = []
        for i in keep:
            row = _design_row(ts[i], f_star, n_harm)
            model = sum(c * r for c, r in zip(coef, row))
            if abs((ys[i] - model) / sigmas[i]) > scan["clip"]:
                drop.append(i)
        if not drop:
            break
        keep = [i for i in keep if i not in set(drop)]

    if len(keep) < scan["min_used"]:
        _fail("剔除离群点后只剩 %d 个有效点，少于要求的 %d" % (len(keep), scan["min_used"]))

    kt = [ts[i] for i in keep]
    ky = [ys[i] for i in keep]
    ks = [sigmas[i] for i in keep]
    lo = max(scan["f_min"], f_star - df)
    hi = min(scan["f_max"], f_star + df)
    if hi <= lo:
        f_best = f_star
    else:
        f_best = _golden(lambda x: _chi2_only(kt, ky, ks, x, n_harm), lo, hi)
    coef, chi2 = fit_at(kt, ky, ks, f_best, n_harm)
    if coef is None:
        _fail("最终拟合解不出来，检查记录")
    return {"f": f_best, "period": 1.0 / f_best, "coef": coef, "chi2": chi2,
            "n_used": len(keep), "n_total": len(ts), "n_harm": n_harm}


def build_report(scan, res):
    n_used = res["n_used"]
    dof = n_used - (2 * res["n_harm"] + 1)
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
        "chi2": round(res["chi2"], 2),
        "reduced_chi2": round(res["chi2"] / dof, 4),
        "meta": {"model": "harmonics-%d" % res["n_harm"], "unit": scan["site"].get("time_system", "HJD")},
    }


def load_cache(path):
    """读复核缓存；读不出来或结构不对就当空缓存。"""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(raw, dict) or raw.get("version") != 1:
        return []
    entries = raw.get("entries")
    if not isinstance(entries, list):
        return []
    out = []
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("key"), str) and isinstance(e.get("report"), dict):
            out.append({"key": e["key"], "report": e["report"]})
    return out


def record_key(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_cache(path, entries):
    atomic_write(path, {"version": 1, "entries": entries})


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
    ap.add_argument("--cache", default=None)
    args = ap.parse_args(argv)

    try:
        key = record_key(args.data)
    except OSError:
        _fail("读不到记录 %s" % args.data)

    entries = load_cache(args.cache) if args.cache else []
    report = None
    for e in entries:
        if e["key"] == key:
            report = e["report"]
            break

    if report is None:
        scan = load_data(args.data)
        res = search_period(scan)
        report = build_report(scan, res)

    if args.cache:
        hit = None
        for i, e in enumerate(entries):
            if e["key"] == key:
                hit = i
                break
        entry = {"key": key, "report": report}
        if hit is None:
            entries = entries + [entry]
        else:
            entries = entries[:]
            entries[hit] = entry
        try:
            save_cache(args.cache, entries)
        except OSError:
            _fail("缓存写不进去：%s" % args.cache)

    try:
        atomic_write(args.out, report)
    except OSError:
        _fail("报告写不进去：%s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
