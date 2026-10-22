#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变星光变曲线：周期复核（骨架版）。

用法：
    python3 src/period.py --data data/lcurve.json --out out/report.json
"""
import argparse
import datetime as dt
import json
import math
import os


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scan_period(raw):
    samples = raw["samples"]
    ts = [float(s["t"]) for s in samples]
    ys = [float(s["dmag"]) for s in samples]
    cfg = raw["fit"]
    f_min = float(cfg["f_min"])
    f_max = float(cfg["f_max"])

    span = max(ts) - min(ts)
    df = 1.0 / (4.0 * span)

    best = None
    f = f_min
    while f <= f_max:
        # 单条正弦：只解常数项与基波的 cos/sin
        sc = ss = scc = sss = scs = syc = sys_ = sy = 0.0
        n = 0.0
        for t, y in zip(ts, ys):
            th = 2.0 * math.pi * f * t
            c = math.cos(th)
            s = math.sin(th)
            n += 1.0
            sc += c
            ss += s
            scc += c * c
            sss += s * s
            scs += c * s
            syc += y * c
            sys_ += y * s
            sy += y
        # 三阶方程：n·c0 + sc·a1 + ss·b1 = sy 这一类
        m = [[n, sc, ss, sy], [sc, scc, scs, syc], [ss, scs, sss, sys_]]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
            if abs(m[piv][col]) < 1e-12:
                m = None
                break
            m[col], m[piv] = m[piv], m[col]
            pv = m[col][col]
            for k in range(col, 4):
                m[col][k] /= pv
            for r in range(3):
                if r != col:
                    fac = m[r][col]
                    for k in range(col, 4):
                        m[r][k] -= fac * m[col][k]
        if m is None:
            f += df
            continue
        c0, a1, b1 = m[0][3], m[1][3], m[2][3]
        chi2 = 0.0
        for t, y in zip(ts, ys):
            th = 2.0 * math.pi * f * t
            model = c0 + a1 * math.cos(th) + b1 * math.sin(th)
            chi2 += (y - model) ** 2
        if best is None or chi2 < best[0]:
            best = (chi2, f, c0, a1, b1)
        f += df

    chi2, f, c0, a1, b1 = best
    return {
        "n_total": len(ts),
        "n_used": len(ts),
        "period_days": round(1.0 / f, 5),
        "amplitude_mmag": round(math.sqrt(a1 * a1 + b1 * b1), 2),
        "chi2": round(chi2, 2),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": {"model": "sine"},
    }


def main():
    ap = argparse.ArgumentParser(description="光变曲线周期搜索")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    raw = load(args.data)
    report = scan_period(raw)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
