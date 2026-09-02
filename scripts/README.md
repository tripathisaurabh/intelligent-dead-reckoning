# Scripts

Repeatable command-line entry points for preprocessing, training, evaluation and plot generation live here.

Avoid one-off scripts in the repository root.

CURRENT IMPLEMENTATION

Timestamp difference:
Δt = t[k] - t[k-1]

Sampling rate:
f = 1 / median(Δt)

Session reset:
Δt <= 0

Large gap:
Δt > chosen threshold

Duration:
T = t[last] - t[first]


NEXT — BASIC INS

Velocity:
v[k] = v[k-1] + a[k]Δt

Position:
p[k] = p[k-1] + v[k]Δt

Orientation:
θ[k] ≈ θ[k-1] + ω[k]Δt