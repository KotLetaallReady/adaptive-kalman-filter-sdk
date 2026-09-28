# benchmarks

Research benchmarks for `core:engine`. Not published. Needs only a JDK.

```bash
./gradlew :benchmarks:run                          # every table
./gradlew :benchmarks:run --args="stage1 summary"  # some of them
./gradlew :benchmarks:run --args="export"          # tracks + dataset CSV
./gradlew :benchmarks:run --args="help"
```

Output: `benchmarks/build/results/<experiment>.txt` (tables) and `<experiment>.csv` (long form: `table, scenario, method, lag_s, metric, value, params`; `lag_s = -1` is a smoother over the whole finished track). Metrics: `rmse`, `rmse_straight`, `rmse_turn`, `rmse_stop`, `p95`, `jitter` (second difference against the truth), `length_err_pct` (path length against the true path), `us_per_step`, plus experiment-specific ones. A Windows console may garble Cyrillic in the Gradle output; the files are UTF-8.

## Method

- 5 scenarios × 60 routes × 600 s at 1 Hz: `WALK` (1.4 m/s) and `BIKE` (5 m/s) with `WHITE` (σ = 4 m), `REALISTIC` (Gauss–Markov drift τ = 30 s + white noise + multipath jumps of 10–40 m + dropouts) and `MISCAL` (open sky and city zones, the phone reports ~4 m in both).
- Phone sensors: Doppler speed and bearing (`speedAccuracy` 0.2 m/s, 0.4 in the city; bearing only above 0.5 m/s), gyro around gravity, accelerometer variance.
- 20 routes tune and train every method, the other 40 test it. The first 16 s of a route are not scored. All methods are scored on the same timestamps.
- The engine runs with `useWallClockTime = false`; network training uses a seeded shuffle, so every run gives the same numbers.
- `EngineParityTest`: the benchmark's own KF-CV matches the SDK `KalmanFilter` point by point, so the competitor filters share one reference.

## Experiments

| Name | What it answers |
|---|---|
| `baseline` | KF vs UKF-CTRV, the SDK α-smoother before and after the turn fix, SG, RTS, oracle α (position only) |
| `turns` | How often turn suppression fires on straight legs, turns and stops |
| `window` | Window length L for SG, the α-smoother and RTS |
| `loss` | Labels and loss weighting of the α-network |
| `leaderboard` | Noise network and "SG only on turns" on top of KF and RTS |
| `noise-input`, `noise-teacher` | Noise estimator inputs and teachers without ground truth (no Doppler) |
| `stage1` | KF / EKF / UKF + RTS with position, + Doppler, + gyro |
| `stage2` | ZUPT rules and classifier, EM calibration, self-trained noise network on KF-CV + Doppler |
| `stage3` | Training on a mix of situations; truth-free acceptance by innovation NLL |
| `stage4` | Robust RTS (IRLS, Huber / Cauchy), online Huber update, robust teacher |
| `gate` | Track loss by the outlier gate on all 60 routes |
| `summary` | Best scheme for real time, a 5 s route line and a finished track, against the SDK as it ships |
| `tracks` | First test route of each scenario, point by point, for plots |
| `dataset` | Every route with truth and all sensors, e.g. for KalmanNet in PyTorch |

Everything here is synthetic: numbers still need checking on real phone logs.

The module compiles as a friend of `core:engine` (`friendPaths` in `build.gradle.kts`), like the engine's own tests, so it can drive `MatrixOps`, `SmootherWindow` and `OptimalAlpha` without making them public API.
