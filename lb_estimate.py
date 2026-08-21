"""Leaderboard-score estimator with submission-based calibration.

The local eval uses the OFFICIAL scoring formula (verified against
docs.arcprize.org/methodology): per-level (baseline/actions)^2 capped at
1.15, level-index weights, max-achievable cap, mean across games. What
differs from the LB is the GAME SET: 110 private games of unknown
difficulty mix vs our 25 public ones.

Model: LB ~= k * local_mean, where k captures the private set's relative
easiness. Each real submission (agent with known local_mean -> LB score)
adds a calibration point; k is estimated from the latest points.

Known calibration points (append after each submission):
  agent      local_mean   LB      note
  v9         ~0.004       0.17    old harness era; k ~= 42 (!) -- the
                                  private set has far more harvestable
                                  easy levels than our locals
  v54        0.1370       ?       (pending: mean over salts 0/1/2 =
                                  0.208/0.098/0.120... conservative:
                                  use salt-mean 0.142)

Usage: python lb_estimate.py <local_mean>
"""
import sys

POINTS = [
    # (local_mean, lb_score)
    (0.004, 0.17),
    # append (0.142, <score>) after the v54 submission lands
]


def estimate(local_mean: float) -> tuple[float, float, float]:
    ks = [lb / lm for (lm, lb) in POINTS if lm > 0]
    k_mid = ks[-1]  # latest point dominates
    # uncertainty: with one point the band is huge; with two+, tighten
    if len(ks) >= 2:
        lo, hi = min(ks), max(ks)
    else:
        lo, hi = k_mid / 8, k_mid  # easy-set advantage shrinks as the
        #                            agent stops relying on luck
    return local_mean * lo, local_mean * k_mid, local_mean * hi


if __name__ == "__main__":
    lm = float(sys.argv[1]) if len(sys.argv) > 1 else 0.142
    lo, mid, hi = estimate(lm)
    print(f"local_mean={lm}")
    print(f"LB estimate: {lo:.2f} .. {mid:.2f} .. {hi:.2f}")
    print("(one calibration point only -- band is honest, not narrow;"
          " append the next submission's score to POINTS)")
