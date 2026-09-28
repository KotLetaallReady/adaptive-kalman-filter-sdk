package com.katka.benchmarks

import com.katka.engine.MatrixOps
import com.katka.model.Observation
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

fun chol(A: Array<DoubleArray>): Array<DoubleArray> {
    val n = A.size; var jitter = 0.0
    repeat(8) {
        val L = Array(n) { DoubleArray(n) }; var ok = true
        loop@ for (i in 0 until n) for (j in 0..i) {
            var s = A[i][j] + if (i == j) jitter else 0.0
            for (k in 0 until j) s -= L[i][k] * L[j][k]
            if (i == j) {
                if (s <= 0.0) { ok = false; break@loop }
                L[i][i] = sqrt(s)
            } else L[i][j] = s / L[j][j]
        }
        if (ok) return L
        jitter = if (jitter == 0.0) 1e-9 else jitter * 100
    }
    error("cholesky failed")
}

// ───────────────────────── EKF / UKF with a CTRV motion model, position only ─────────────────────────

/**
 * Coordinated turn with constant velocity, state [x, y, v, ψ, ω]. Uses the SDK's gate rule;
 * [gateRecovery] accepts a fix after 3 rejections in a row, like the SDK since step 1.
 */
class Ctrv(private val sa: Double, private val sw: Double, private val unscented: Boolean, private val gateRecovery: Boolean = true) {
    private var x = DoubleArray(5)
    private var P = MatrixOps.zeros(5, 5)

    fun run(obs: List<Observation>): Track {
        val out = HashMap<Long, V2>()
        var first: V2? = null; var lastT = 0L; var stage = 0; var streak = 0
        for (o in obs) {
            val z = toLocal(o.latitude, o.longitude)
            val acc = o.accuracy.toDouble().coerceIn(1.0, 50.0); val r = acc * acc
            if (stage == 0) { first = z; lastT = o.timestamp; stage = 1; out[o.timestamp] = z; continue }
            val dt = ((o.timestamp - lastT) / 1000.0).coerceIn(0.001, 30.0); lastT = o.timestamp
            if (stage == 1) {
                val d = z - first!!
                x = doubleArrayOf(z.x, z.y, d.norm() / dt, atan2(d.y, d.x), 0.0)
                P = MatrixOps.diagonal(doubleArrayOf(r, r, 4.0, (PI / 2) * (PI / 2), 0.25))
                stage = 2; out[o.timestamp] = z; continue
            }
            if (unscented) predictUkf(dt) else predictEkf(dt)
            val iv = doubleArrayOf(z.x - x[0], z.y - x[1])
            val gate = (sqrt(P[0][0] + P[1][1]) * 5.0).coerceIn(15.0, 80.0)   // same gate rule as the SDK
            if (hypot(iv[0], iv[1]) <= gate || (gateRecovery && streak >= 3)) {
                update(iv, r); streak = 0
            } else streak++
            out[o.timestamp] = V2(x[0], x[1])
        }
        return out
    }

    private fun f(s: DoubleArray, dt: Double): DoubleArray {
        val px = s[0]; val py = s[1]; val v = s[2]; val psi = s[3]; val w = s[4]
        return if (abs(w) > 1e-4) doubleArrayOf(
            px + v / w * (sin(psi + w * dt) - sin(psi)),
            py + v / w * (cos(psi) - cos(psi + w * dt)),
            v, psi + w * dt, w
        ) else doubleArrayOf(
            px + v * cos(psi) * dt - 0.5 * v * w * dt * dt * sin(psi),
            py + v * sin(psi) * dt + 0.5 * v * w * dt * dt * cos(psi),
            v, psi + w * dt, w
        )
    }

    private fun q(psi: Double, dt: Double): Array<DoubleArray> {
        val h = 0.5 * dt * dt
        val G = arrayOf(
            doubleArrayOf(h * cos(psi), 0.0), doubleArrayOf(h * sin(psi), 0.0),
            doubleArrayOf(dt, 0.0), doubleArrayOf(0.0, h), doubleArrayOf(0.0, dt)
        )
        val D = MatrixOps.diagonal(doubleArrayOf(sa * sa, sw * sw))
        return MatrixOps.addDiagEps(MatrixOps.mul(MatrixOps.mul(G, D), MatrixOps.transpose(G)), 1e-9)
    }

    private fun predictEkf(dt: Double) {
        val F = Array(5) { DoubleArray(5) }; val h = 1e-6
        for (j in 0 until 5) {
            val xp = x.copyOf(); xp[j] += h; val xm = x.copyOf(); xm[j] -= h
            val fp = f(xp, dt); val fm = f(xm, dt)
            for (i in 0 until 5) F[i][j] = (fp[i] - fm[i]) / (2 * h)
        }
        val Q = q(x[3], dt)
        x = f(x, dt); x[3] = wrap(x[3])
        P = MatrixOps.symmetrise(MatrixOps.add(MatrixOps.mul(MatrixOps.mul(F, P), MatrixOps.transpose(F)), Q))
    }

    private fun predictUkf(dt: Double) {
        val n = 5; val lam = 0.0; val c = sqrt(n + lam)       // Merwe: alpha=1, beta=2, kappa=0
        val L = chol(P)
        val sig = Array(2 * n + 1) { x.copyOf() }
        for (i in 0 until n) for (r in 0 until n) { sig[1 + i][r] += c * L[r][i]; sig[1 + n + i][r] -= c * L[r][i] }
        val wm = DoubleArray(2 * n + 1) { if (it == 0) lam / (n + lam) else 1.0 / (2 * (n + lam)) }
        val wc = wm.copyOf().also { it[0] += 2.0 }
        val prop = sig.map { f(it, dt) }
        val m = DoubleArray(n); var sn = 0.0; var cs = 0.0
        for (k in prop.indices) {
            for (i in 0 until n) if (i != 3) m[i] += wm[k] * prop[k][i]
            sn += wm[k] * sin(prop[k][3]); cs += wm[k] * cos(prop[k][3])
        }
        m[3] = atan2(sn, cs)
        val Pn = MatrixOps.zeros(n, n)
        for (k in prop.indices) {
            val d = DoubleArray(n) { prop[k][it] - m[it] }; d[3] = wrap(d[3])
            for (i in 0 until n) for (j in 0 until n) Pn[i][j] += wc[k] * d[i] * d[j]
        }
        x = m; P = MatrixOps.symmetrise(MatrixOps.add(Pn, q(m[3], dt)))
    }

    private fun update(iv: DoubleArray, r: Double) {
        val S = arrayOf(doubleArrayOf(P[0][0] + r, P[0][1]), doubleArrayOf(P[1][0], P[1][1] + r))
        val Si = MatrixOps.inverse(S)
        val K = Array(5) { i -> DoubleArray(2) { j -> P[i][0] * Si[0][j] + P[i][1] * Si[1][j] } }
        for (i in 0 until 5) x[i] += K[i][0] * iv[0] + K[i][1] * iv[1]
        x[3] = wrap(x[3])
        val IKH = MatrixOps.identity(5)
        for (i in 0 until 5) { IKH[i][0] -= K[i][0]; IKH[i][1] -= K[i][1] }
        val R = MatrixOps.diagonal(doubleArrayOf(r, r))
        P = MatrixOps.symmetrise(MatrixOps.add(
            MatrixOps.mul(MatrixOps.mul(IKH, P), MatrixOps.transpose(IKH)),
            MatrixOps.mul(MatrixOps.mul(K, R), MatrixOps.transpose(K))
        ))
    }
}

// ───────────────────────── one Gaussian filter for CV / CTRV, linearised or unscented ─────────────────────────

enum class Motion { CV, CTRV }

/** Filtered state plus what the RTS pass needs: prediction and cross-covariance D = Cov(x_{k-1}, x_k | k-1). */
class GStep(val ts: Long, val xf: DoubleArray, val pf: Array<DoubleArray>,
            val xp: DoubleArray?, val pp: Array<DoubleArray>?, val d: Array<DoubleArray>?)

/**
 * KF-CV / EKF-CTRV / UKF-CTRV on position, optionally with Doppler speed and bearing, gyro, ZUPT and a per-step R.
 * Position updates use the SDK gate with recovery after 3 rejections.
 */
class GFilter(
    val motion: Motion, val unscented: Boolean, val qa: Double, val qw: Double,
    val doppler: Boolean, val gyro: Boolean,
    val rFn: RModel? = null,
    val stillFn: ((Meas) -> Boolean)? = null,
    val onStep: ((RCtx, Meas) -> Unit)? = null
) {
    val n = if (motion == Motion.CV) 4 else 5
    private val psiIdx = if (motion == Motion.CTRV) 3 else -1
    private var x = DoubleArray(n)
    private var P = MatrixOps.zeros(n, n)
    private var lastD: Array<DoubleArray>? = null

    /** Innovation negative log-likelihood of the position updates: a truth-free score of how well R fits. */
    var nll = 0.0; var nllN = 0

    fun run(ms: List<Meas>): List<GStep> {
        val out = ArrayList<GStep>(ms.size)
        var lastT = 0L; var stage = 0; var first: V2? = null; var streak = 0
        val hist = ArrayDeque<Double>()
        for (m in ms) {
            val o = m.obs
            val z = toLocal(o.latitude, o.longitude)
            val acc = o.accuracy.toDouble().coerceIn(1.0, 50.0)
            if (stage < 2) {
                val dt = if (stage == 0) 0.0 else ((o.timestamp - lastT) / 1000.0).coerceIn(0.001, 30.0)
                lastT = o.timestamp
                when {
                    motion == Motion.CV -> {
                        x = doubleArrayOf(z.x, z.y, 0.0, 0.0); P = MatrixOps.diagonal(doubleArrayOf(acc * acc, acc * acc, 10.0, 10.0)); stage = 2
                    }
                    doppler && o.hasBearing -> {
                        val b = Math.toRadians(m.bearingAccDeg)
                        x = doubleArrayOf(z.x, z.y, o.speed.toDouble(), bearingToPsi(o.bearing), 0.0)
                        P = MatrixOps.diagonal(doubleArrayOf(acc * acc, acc * acc, 1.0, b * b + 0.01, 0.25)); stage = 2
                    }
                    stage == 1 -> {
                        val d = z - first!!
                        x = doubleArrayOf(z.x, z.y, d.norm() / dt, atan2(d.y, d.x), 0.0)
                        P = MatrixOps.diagonal(doubleArrayOf(acc * acc, acc * acc, 4.0, (PI / 2) * (PI / 2), 0.25)); stage = 2
                    }
                    else -> { first = z; stage = 1; x = DoubleArray(n).also { it[0] = z.x; it[1] = z.y } }
                }
                out += GStep(o.timestamp, x.copyOf(), MatrixOps.copy(P), null, null, null)
                continue
            }
            val dt = ((o.timestamp - lastT) / 1000.0).coerceIn(0.001, 30.0); lastT = o.timestamp
            predict(dt)
            val xp = x.copyOf(); val pp = MatrixOps.copy(P); val d = lastD

            // position, with the SDK's gate (+ recovery after 3 rejects) and R from accuracy or a model
            val nu = hypot(z.x - x[0], z.y - x[1])
            val mean = if (hist.isEmpty()) nu else hist.average()
            val std = if (hist.size < 2) 0.0 else sqrt(hist.sumOf { (it - mean) * (it - mean) } / hist.size)
            val ctx = RCtx(o, nu, mean, std, sqrt(P[0][0] + P[1][1]))
            val r = rFn?.invoke(ctx)?.coerceIn(1.0, 2500.0) ?: (acc * acc)
            onStep?.invoke(ctx, m)
            val gate = (sqrt(P[0][0] + P[1][1]) * 5.0).coerceIn(15.0, 80.0)
            if (nu <= gate || streak >= 3) {
                val s00 = P[0][0] + r; val s11 = P[1][1] + r; val s01 = P[0][1]; val det = s00 * s11 - s01 * s01
                val dx = z.x - x[0]; val dy = z.y - x[1]
                nll += ln(det) + (s11 * dx * dx - 2 * s01 * dx * dy + s00 * dy * dy) / det; nllN++
                update(intArrayOf(0, 1), doubleArrayOf(z.x, z.y), MatrixOps.diagonal(doubleArrayOf(r, r))); streak = 0
            }
            else streak++
            hist.addLast(nu); if (hist.size > 10) hist.removeFirst()

            if (doppler) dopplerUpdate(m)
            if (gyro && motion == Motion.CTRV) update(intArrayOf(4), doubleArrayOf(m.gyroZ), arrayOf(doubleArrayOf(0.05 * 0.05)))
            if (stillFn?.invoke(m) == true) zupt()
            out += GStep(o.timestamp, x.copyOf(), MatrixOps.copy(P), xp, pp, d)
        }
        return out
    }

    private fun dopplerUpdate(m: Meas) {
        val o = m.obs; val s = o.speed.toDouble(); val sa = m.speedAcc
        if (motion == Motion.CTRV) {
            update(intArrayOf(2), doubleArrayOf(s), arrayOf(doubleArrayOf(sa * sa)))
            if (o.hasBearing) {
                val b = Math.toRadians(m.bearingAccDeg)
                update(intArrayOf(3), doubleArrayOf(bearingToPsi(o.bearing)), arrayOf(doubleArrayOf(b * b)))
            }
        } else if (o.hasBearing) {
            // speed + bearing → velocity vector, covariance through the Jacobian of the polar→cartesian map
            val psi = bearingToPsi(o.bearing); val b = Math.toRadians(m.bearingAccDeg)
            val J = arrayOf(doubleArrayOf(cos(psi), -s * sin(psi)), doubleArrayOf(sin(psi), s * cos(psi)))
            val R = MatrixOps.mul(MatrixOps.mul(J, MatrixOps.diagonal(doubleArrayOf(sa * sa, b * b))), MatrixOps.transpose(J))
            update(intArrayOf(2, 3), doubleArrayOf(s * cos(psi), s * sin(psi)), MatrixOps.addDiagEps(R, 1e-6))
        } else {
            // slow: direction unknown, but the speed itself bounds the velocity
            val v = s * s + sa * sa
            update(intArrayOf(2, 3), doubleArrayOf(0.0, 0.0), MatrixOps.diagonal(doubleArrayOf(v, v)))
        }
    }

    /** Zero-velocity update: standing still is a measurement, not a guess. */
    private fun zupt() {
        if (motion == Motion.CTRV) {
            update(intArrayOf(2), doubleArrayOf(0.0), arrayOf(doubleArrayOf(0.05 * 0.05)))
            update(intArrayOf(4), doubleArrayOf(0.0), arrayOf(doubleArrayOf(0.02 * 0.02)))
        } else update(intArrayOf(2, 3), doubleArrayOf(0.0, 0.0), MatrixOps.diagonal(doubleArrayOf(0.0025, 0.0025)))
    }

    /** Linear update of the selected state components (every measurement here is a component pick). */
    private fun update(idx: IntArray, z: DoubleArray, R: Array<DoubleArray>) {
        val m = idx.size
        val y = DoubleArray(m) { z[it] - x[idx[it]] }
        for (i in 0 until m) if (idx[i] == psiIdx) y[i] = wrap(y[i])
        val S = Array(m) { i -> DoubleArray(m) { j -> P[idx[i]][idx[j]] + R[i][j] } }
        val Si = MatrixOps.inverse(S)
        val K = Array(n) { r -> DoubleArray(m) { c -> (0 until m).sumOf { k -> P[r][idx[k]] * Si[k][c] } } }
        for (r in 0 until n) x[r] += (0 until m).sumOf { k -> K[r][k] * y[k] }
        if (psiIdx >= 0) x[psiIdx] = wrap(x[psiIdx])
        val IKH = MatrixOps.identity(n)
        for (r in 0 until n) for (k in 0 until m) IKH[r][idx[k]] -= K[r][k]
        P = MatrixOps.symmetrise(MatrixOps.add(
            MatrixOps.mul(MatrixOps.mul(IKH, P), MatrixOps.transpose(IKH)),
            MatrixOps.mul(MatrixOps.mul(K, R), MatrixOps.transpose(K))
        ))
    }

    // ── motion models ──

    private fun ctrv(s: DoubleArray, dt: Double): DoubleArray {
        val px = s[0]; val py = s[1]; val v = s[2]; val psi = s[3]; val w = s[4]
        return if (abs(w) > 1e-4) doubleArrayOf(
            px + v / w * (sin(psi + w * dt) - sin(psi)), py + v / w * (cos(psi) - cos(psi + w * dt)), v, psi + w * dt, w
        ) else doubleArrayOf(
            px + v * cos(psi) * dt - 0.5 * v * w * dt * dt * sin(psi), py + v * sin(psi) * dt + 0.5 * v * w * dt * dt * cos(psi), v, psi + w * dt, w
        )
    }

    private fun qCtrv(psi: Double, dt: Double): Array<DoubleArray> {
        val h = 0.5 * dt * dt
        val G = arrayOf(doubleArrayOf(h * cos(psi), 0.0), doubleArrayOf(h * sin(psi), 0.0), doubleArrayOf(dt, 0.0), doubleArrayOf(0.0, h), doubleArrayOf(0.0, dt))
        return MatrixOps.addDiagEps(MatrixOps.mul(MatrixOps.mul(G, MatrixOps.diagonal(doubleArrayOf(qa * qa, qw * qw))), MatrixOps.transpose(G)), 1e-9)
    }

    private fun predict(dt: Double) {
        when {
            motion == Motion.CV -> {
                val F = arrayOf(doubleArrayOf(1.0, 0.0, dt, 0.0), doubleArrayOf(0.0, 1.0, 0.0, dt), doubleArrayOf(0.0, 0.0, 1.0, 0.0), doubleArrayOf(0.0, 0.0, 0.0, 1.0))
                val q = qa * qa; val d2 = dt * dt; val d3 = d2 * dt; val d4 = d3 * dt
                val Q = arrayOf(
                    doubleArrayOf(q * d4 / 4, 0.0, q * d3 / 2, 0.0), doubleArrayOf(0.0, q * d4 / 4, 0.0, q * d3 / 2),
                    doubleArrayOf(q * d3 / 2, 0.0, q * d2, 0.0), doubleArrayOf(0.0, q * d3 / 2, 0.0, q * d2)
                )
                lastD = MatrixOps.mul(P, MatrixOps.transpose(F))
                x = MatrixOps.mulVec(F, x)
                P = MatrixOps.symmetrise(MatrixOps.add(MatrixOps.mul(F, lastD!!), Q))
            }
            !unscented -> {
                val F = Array(5) { DoubleArray(5) }; val h = 1e-6
                for (j in 0 until 5) {
                    val a = x.copyOf(); a[j] += h; val b = x.copyOf(); b[j] -= h
                    val fa = ctrv(a, dt); val fb = ctrv(b, dt)
                    for (i in 0 until 5) F[i][j] = (fa[i] - fb[i]) / (2 * h)
                }
                val Q = qCtrv(x[3], dt)
                lastD = MatrixOps.mul(P, MatrixOps.transpose(F))
                x = ctrv(x, dt); x[3] = wrap(x[3])
                P = MatrixOps.symmetrise(MatrixOps.add(MatrixOps.mul(F, lastD!!), Q))
            }
            else -> {
                val nn = 5; val lam = 0.0; val c = sqrt(nn + lam)
                val L = chol(P)
                val sig = Array(2 * nn + 1) { x.copyOf() }
                for (i in 0 until nn) for (r in 0 until nn) { sig[1 + i][r] += c * L[r][i]; sig[1 + nn + i][r] -= c * L[r][i] }
                val wm = DoubleArray(2 * nn + 1) { if (it == 0) lam / (nn + lam) else 1.0 / (2 * (nn + lam)) }
                val wc = wm.copyOf().also { it[0] += 2.0 }
                val prop = sig.map { ctrv(it, dt) }
                val mu = DoubleArray(nn); var sn = 0.0; var cs = 0.0
                for (k in prop.indices) {
                    for (i in 0 until nn) if (i != 3) mu[i] += wm[k] * prop[k][i]
                    sn += wm[k] * sin(prop[k][3]); cs += wm[k] * cos(prop[k][3])
                }
                mu[3] = atan2(sn, cs)
                val Pn = MatrixOps.zeros(nn, nn); val D = MatrixOps.zeros(nn, nn)
                for (k in prop.indices) {
                    val dy = DoubleArray(nn) { prop[k][it] - mu[it] }; dy[3] = wrap(dy[3])
                    val dx = DoubleArray(nn) { sig[k][it] - x[it] }; dx[3] = wrap(dx[3])
                    for (i in 0 until nn) for (j in 0 until nn) { Pn[i][j] += wc[k] * dy[i] * dy[j]; D[i][j] += wc[k] * dx[i] * dy[j] }
                }
                lastD = D; x = mu
                P = MatrixOps.symmetrise(MatrixOps.add(Pn, qCtrv(mu[3], dt)))
            }
        }
    }
}

// ───────────────────────── smoothing over GStep ─────────────────────────

fun filtTrack(steps: List<GStep>): Track = steps.associate { it.ts to V2(it.xf[0], it.xf[1]) }

/** Fixed-lag RTS / ERTS / URTS: C_i = D_{i+1} · P_{i+1|i}^-1, the same formula for all three filters. */
fun fixedLag(steps: List<GStep>, lag: Int, psiIdx: Int): Track {
    val out = HashMap<Long, V2>()
    for (k in lag until steps.size) {
        val j = k - lag
        var xs = steps[k].xf; var ok = true
        for (i in k - 1 downTo j) {
            val nx = steps[i + 1]
            if (nx.xp == null) { ok = false; break }
            val C = MatrixOps.mul(nx.d!!, MatrixOps.inverse(nx.pp!!))
            val diff = DoubleArray(xs.size) { xs[it] - nx.xp[it] }
            if (psiIdx >= 0) diff[psiIdx] = wrap(diff[psiIdx])
            val cd = MatrixOps.mulVec(C, diff)
            xs = DoubleArray(xs.size) { steps[i].xf[it] + cd[it] }
        }
        val p = if (ok) xs else steps[j].xf
        out[steps[j].ts] = V2(p[0], p[1])
    }
    return out
}

/** Full offline RTS with covariances (the teacher for self-training). */
fun fullSmooth(steps: List<GStep>, psiIdx: Int): Pair<Array<DoubleArray>, Array<Array<DoubleArray>>> {
    val n = steps.size
    val xs = Array(n) { steps[it].xf.copyOf() }; val ps = Array(n) { steps[it].pf }
    for (i in n - 2 downTo 0) {
        val nx = steps[i + 1]
        if (nx.xp == null) continue
        val C = MatrixOps.mul(nx.d!!, MatrixOps.inverse(nx.pp!!))
        val diff = DoubleArray(xs[i].size) { xs[i + 1][it] - nx.xp[it] }
        if (psiIdx >= 0) diff[psiIdx] = wrap(diff[psiIdx])
        val cd = MatrixOps.mulVec(C, diff)
        xs[i] = DoubleArray(xs[i].size) { steps[i].xf[it] + cd[it] }
        ps[i] = MatrixOps.add(steps[i].pf, MatrixOps.mul(MatrixOps.mul(C, MatrixOps.sub(ps[i + 1], nx.pp)), MatrixOps.transpose(C)))
    }
    return xs to ps
}

data class Variant(val name: String, val motion: Motion, val ukf: Boolean, val doppler: Boolean, val gyro: Boolean)

val STAGE1 = listOf(
    Variant("KF-CV", Motion.CV, false, false, false),
    Variant("EKF-CTRV", Motion.CTRV, false, false, false),
    Variant("UKF-CTRV", Motion.CTRV, true, false, false),
    Variant("KF-CV + Doppler", Motion.CV, false, true, false),
    Variant("EKF-CTRV + Doppler", Motion.CTRV, false, true, false),
    Variant("UKF-CTRV + Doppler", Motion.CTRV, true, true, false),
    Variant("EKF-CTRV + Doppler + гиро", Motion.CTRV, false, true, true),
    Variant("UKF-CTRV + Doppler + гиро", Motion.CTRV, true, true, true)
)

/** The base found in stage 1: KF-CV + Doppler. */
val KF_DOPPLER = STAGE1[3]

/** Fixed-lag delay in samples (1 Hz), the same 5 s as the SDK smoother window L = 11. */
const val LAG = 5
