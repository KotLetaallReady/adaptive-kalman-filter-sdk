package com.katka.benchmarks

import com.katka.engine.KalmanFilter
import com.katka.engine.MatrixOps
import com.katka.engine.coefficient_startegy.ClassicalCoefficientStrategy
import com.katka.engine.coefficient_startegy.CoefficientStrategy
import com.katka.engine.model.FilterResult
import com.katka.engine.model.GainResult
import com.katka.engine.smoothing.SmootherInput
import com.katka.model.Observation
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.sqrt

// ───────────────────────── the SDK KalmanFilter on recorded observations ─────────────────────────

/** One engine run: the filter (for its local frame), the observations and every step result. */
class KfTrace(val f: KalmanFilter, val obs: List<Observation>, val res: List<FilterResult>) {
    fun toMy(x: Double, y: Double): V2 { val (la, lo) = f.localToGeo(x, y); return toLocal(la, lo) }
    fun toEngine(p: V2): DoubleArray { val (la, lo) = toGeo(p); val (x, y) = f.geoToLocal(la, lo); return doubleArrayOf(x, y) }
    fun track(): Track = res.indices.associate { obs[it].timestamp to toMy(res[it].state.x, res[it].state.y) }
    fun input(i: Int): SmootherInput {
        val r = res[i]; val o = obs[i]
        val (rx, ry) = f.geoToLocal(o.latitude, o.longitude)
        return SmootherInput(
            timestamp = o.timestamp, kfX = r.state.x, kfY = r.state.y, rawX = rx, rawY = ry,
            vx = r.state.vx, vy = r.state.vy,
            speed = if (o.hasSpeed) o.speed.toDouble() else hypot(r.state.vx, r.state.vy),
            accuracy = o.accuracy.toDouble(),
            innovationMag = hypot(r.innovation[0], r.innovation[1]),
            sigmaPos = r.state.positionUncertaintyMeters
        )
    }
}

/** The SDK filter with R from accuracy (optionally Sage–Husa), exactly as the facade builds it, but timestamp-only dt. */
fun runKf(obs: List<Observation>, q: Double, adaptive: Boolean): KfTrace {
    val f = KalmanFilter(q, useWallClockTime = false); val s = ClassicalCoefficientStrategy(adaptiveR = adaptive)
    f.reset(s)
    val res = ArrayList<FilterResult>(obs.size)
    for (o in obs) res += f.process(o, s)
    return KfTrace(f, obs, res)
}

/** Fixed-lag RTS on top of the SDK KF, in the engine frame, indexed by the smoothed (centre) sample. */
fun rtsEngine(tr: KfTrace, lag: Int): Array<DoubleArray?> {
    val out = arrayOfNulls<DoubleArray>(tr.res.size)
    for (k in lag until tr.res.size) {
        var xs = tr.res[k].state.toVector()
        for (i in k - 1 downTo k - lag) {
            val nx = tr.res[i + 1]; val dt = nx.dt
            val F = arrayOf(
                doubleArrayOf(1.0, 0.0, dt, 0.0), doubleArrayOf(0.0, 1.0, 0.0, dt),
                doubleArrayOf(0.0, 0.0, 1.0, 0.0), doubleArrayOf(0.0, 0.0, 0.0, 1.0)
            )
            val C = MatrixOps.mul(MatrixOps.mul(tr.res[i].state.P, MatrixOps.transpose(F)), MatrixOps.inverse(nx.predicted.P))
            val xp = nx.predicted.toVector()
            val cd = MatrixOps.mulVec(C, DoubleArray(4) { xs[it] - xp[it] })
            val xf = tr.res[i].state.toVector()
            xs = DoubleArray(4) { xf[it] + cd[it] }
        }
        out[k - lag] = xs
    }
    return out
}

fun rtsTrack(tr: KfTrace, rts: Array<DoubleArray?>): Track =
    rts.indices.filter { rts[it] != null }.associate { tr.obs[it].timestamp to tr.toMy(rts[it]!![0], rts[it]!![1]) }

/** Offline RTS over the whole recorded track: smoothed states and covariances (engine frame). */
fun rtsFull(tr: KfTrace): Pair<Array<DoubleArray>, Array<Array<DoubleArray>>> {
    val n = tr.res.size
    val xs = Array(n) { DoubleArray(4) }; val ps = Array(n) { MatrixOps.zeros(4, 4) }
    xs[n - 1] = tr.res[n - 1].state.toVector(); ps[n - 1] = tr.res[n - 1].state.P
    for (i in n - 2 downTo 0) {
        val nx = tr.res[i + 1]; val dt = nx.dt
        val F = arrayOf(
            doubleArrayOf(1.0, 0.0, dt, 0.0), doubleArrayOf(0.0, 1.0, 0.0, dt),
            doubleArrayOf(0.0, 0.0, 1.0, 0.0), doubleArrayOf(0.0, 0.0, 0.0, 1.0)
        )
        val Pf = tr.res[i].state.P; val Pp = nx.predicted.P
        val C = MatrixOps.mul(MatrixOps.mul(Pf, MatrixOps.transpose(F)), MatrixOps.inverse(Pp))
        val xp = nx.predicted.toVector(); val xf = tr.res[i].state.toVector()
        val cd = MatrixOps.mulVec(C, DoubleArray(4) { xs[i + 1][it] - xp[it] })
        xs[i] = DoubleArray(4) { xf[it] + cd[it] }
        ps[i] = MatrixOps.add(Pf, MatrixOps.mul(MatrixOps.mul(C, MatrixOps.sub(ps[i + 1], Pp)), MatrixOps.transpose(C)))
    }
    return xs to ps
}

// ───────────────────────── the SDK filter with R chosen per step ─────────────────────────

/** Kalman update with an externally chosen per-axis variance [r] (Joseph form, like the SDK strategy). */
class ProvidedRStrategy : CoefficientStrategy {
    var r = 25.0
    override fun computeGain(P_pred: Array<DoubleArray>, H: Array<DoubleArray>, obs: Observation): GainResult {
        val R = MatrixOps.diagonal(doubleArrayOf(r, r))
        val PHt = MatrixOps.mul(P_pred, MatrixOps.transpose(H))
        val S = MatrixOps.add(MatrixOps.mul(H, PHt), R)
        val K = MatrixOps.mul(PHt, MatrixOps.inverse(S))
        val IKH = MatrixOps.sub(MatrixOps.identity(P_pred.size), MatrixOps.mul(K, H))
        val P = MatrixOps.symmetrise(MatrixOps.add(
            MatrixOps.mul(MatrixOps.mul(IKH, P_pred), MatrixOps.transpose(IKH)),
            MatrixOps.mul(MatrixOps.mul(K, R), MatrixOps.transpose(K))
        ))
        return GainResult(K, P, R)
    }
}

/** Everything known before the update at this step (causal). */
class RCtx(val obs: Observation, val nu: Double, val nuMean: Double, val nuStd: Double, val sigmaPos: Double)

typealias RModel = (RCtx) -> Double

/**
 * Noise estimator inputs: ln(accuracy), mean and std of |innovation| over the last 10 steps,
 * speed and ln σ_pos. [withNu] adds the current innovation, which made the estimator unstable.
 */
fun rFeatures(c: RCtx, withNu: Boolean): DoubleArray {
    val base = doubleArrayOf(ln(c.obs.accuracy.toDouble()), ln(1 + c.nuMean), ln(1 + c.nuStd), c.obs.speed.toDouble(), ln(c.sigmaPos))
    return if (withNu) base + ln(1 + c.nu) else base
}

/** Drives the SDK KalmanFilter with R chosen per step by [rFn]. */
fun runKfR(obs: List<Observation>, q: Double, rFn: RModel, onStep: ((RCtx, Observation) -> Unit)? = null): KfTrace {
    val f = KalmanFilter(q, useWallClockTime = false); val s = ProvidedRStrategy(); f.reset(s)
    val res = ArrayList<FilterResult>(obs.size); val hist = ArrayDeque<Double>()
    var lastTs = 0L
    for (o in obs) {
        val st = f.getCurrentState()
        if (st != null) {
            val dt = ((o.timestamp - lastTs) / 1000.0).coerceIn(0.001, 30.0)
            val (zx, zy) = f.geoToLocal(o.latitude, o.longitude)
            val nu = hypot(zx - (st.x + st.vx * dt), zy - (st.y + st.vy * dt))
            val mean = if (hist.isEmpty()) nu else hist.average()
            val std = if (hist.size < 2) 0.0 else sqrt(hist.sumOf { (it - mean) * (it - mean) } / hist.size)
            val ctx = RCtx(o, nu, mean, std, st.positionUncertaintyMeters)
            s.r = rFn(ctx).coerceIn(1.0, 2500.0)
            onStep?.invoke(ctx, o)
        }
        val r = f.process(o, s)
        if (st != null) { hist.addLast(hypot(r.innovation[0], r.innovation[1])); if (hist.size > 10) hist.removeFirst() }
        res += r; lastTs = o.timestamp
    }
    return KfTrace(f, obs, res)
}

/** R = accuracy², clamped like ClassicalCoefficientStrategy. */
val accuracyR: RModel = { c -> val a = c.obs.accuracy.toDouble().coerceIn(1.0, 50.0); a * a }

/** R from the true noise level of the zone (simulator side channel): the ceiling for any noise estimator. */
val oracleR: RModel = { c -> c.obs.altitude * c.obs.altitude }
