#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成变星光变曲线（三阶谐波 + 观测噪声 + 宇宙线尖峰 + 观测夜窗口）。

数据口径：业务模型 + 确定性生成规则（固定伪随机序列，不用真随机）——
  相位 th = 2π(t mod P)/P，差分星等 dmag = c0 + Σ_k [a_k cos(k·th) + b_k sin(k·th)]；
  观测只在夜里做，每个观测夜占 8 小时窗口内均匀取点，夜与夜之间留白天间隔；
  噪声按 sigma 抽；另外随机挑几个点叠加 8 倍 sigma 以上的尖峰（宇宙线打点）。
"""
import json
import math
import sys

HARMONICS = ((1, 42.0, 11.0), (2, -14.0, 7.0), (3, 5.0, -3.5))


class Lcg(object):
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def next_u(self):
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return (self.state % 1000003) / 1000003.0


def make_curve(period=1.8539, span_days=120.0, n_nights=40, per_night=10,
               sigma=20.0, harmonics=HARMONICS, n_spikes=9, seed=771003):
    rng = Lcg(seed)
    step = span_days / n_nights
    samples = []
    for night in range(n_nights):
        start = round(night * step + rng.next_u() * 1.2, 5)
        for _ in range(per_night):
            t = round(start + rng.next_u() * 0.34, 5)
            th = 2.0 * math.pi * (t % period) / period
            m = sum(a * math.cos(k * th) + b * math.sin(k * th) for k, a, b in harmonics)
            m += (rng.next_u() + rng.next_u() - 1.0) * math.sqrt(6.0) * sigma
            samples.append({"t": t, "dmag": round(m, 3), "sigma": sigma})
    # 宇宙线尖峰：随机挑几个点往上打
    for _ in range(n_spikes):
        idx = int(rng.next_u() * len(samples)) % len(samples)
        spike = (3.5 + 6.0 * rng.next_u()) * sigma
        samples[idx]["dmag"] = round(samples[idx]["dmag"] + spike, 3)
    samples.sort(key=lambda s: s["t"])
    return {
        "site": {"telescope": "b2", "band": "V", "time_system": "HJD"},
        "fit": {"f_min": 0.02, "f_max": 2.0, "n_harmonics": 3,
                "sigma_clip": 3.0, "max_clip_iter": 2, "min_used": 100},
        "samples": samples,
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "lcurve.json"
    data = make_curve()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print("wrote %s  samples=%d" % (out, len(data["samples"])))
