package com.katka.benchmarks

import com.katka.engine.neural.NetworkConfig
import com.katka.engine.neural.NeuralNetwork
import com.katka.engine.neural.NeuralNetworkTrainer
import com.katka.engine.neural.OutputActivation
import com.katka.engine.neural.TrainingSample
import com.katka.engine.smoothing.FeatureNormalizer
import com.katka.model.Observation
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.random.Random

/** E[ln χ²₂]: converts a predicted ln|z − x|² into a per-axis variance. */
private const val LN_CHI2_2 = 0.1159

/** MLP 5→16→8→1 fitted to ln targets; returns R(ctx) = exp(prediction − E[ln χ²₂]). */
private fun fitLogR(rows: List<DoubleArray>, ys: List<Double>, withNu: Boolean): RModel {
    val norm = FeatureNormalizer.fit(rows); val yMean = ys.average()
    val net = NeuralNetwork(NetworkConfig(rows[0].size, listOf(16, 8), 1, OutputActivation.LINEAR), seed = 7L)
    NeuralNetworkTrainer(net, random = Random(7L)).train(rows.indices.map { TrainingSample(norm.normalize(rows[it]), doubleArrayOf(ys[it] - yMean)) }, epochs = 40)
    return { c -> exp(net.predict(norm.normalize(rFeatures(c, withNu)))[0] + yMean - LN_CHI2_2) }
}

// ───────────────────────── noise estimator on the SDK KF (no Doppler) ─────────────────────────

/**
 * Learns ln(|z − truth|²) from causal features. Two rounds: the second one is trained on
 * innovations produced by the learned-R filter itself.
 */
fun learnR(train: List<Pair<Truth, List<Observation>>>, q: Double, withNu: Boolean): RModel {
    val rows = ArrayList<DoubleArray>(); val ys = ArrayList<Double>()
    var rFn = accuracyR
    repeat(2) {
        for ((t, o) in train) runKfR(o, q, rFn) { ctx, ob ->
            val e = (toLocal(ob.latitude, ob.longitude) - t.pos[t.idx.getValue(ob.timestamp)]).norm()
            rows += rFeatures(ctx, withNu); ys += ln(e * e + 0.01)
        }
        rFn = fitLogR(rows, ys, withNu)
    }
    return rFn
}

/** Who plays "ground truth" when the noise network is trained. */
enum class Teacher { TRUTH, SG, RTS, RTS_EM }

/** Squared measurement residual per timestamp, as seen by the chosen teacher. */
internal fun teacherTargets(teacher: Teacher, truth: Truth, tr: KfTrace): Map<Long, Double> = when (teacher) {
    Teacher.TRUTH -> tr.obs.associate { o ->
        val e = (toLocal(o.latitude, o.longitude) - truth.pos[truth.idx.getValue(o.timestamp)]).norm(); o.timestamp to e * e
    }
    Teacher.SG -> HashMap<Long, Double>().also { m ->
        walk(tr, CFG_NEW, Base.KF, null) { _, c, _, sg ->
            val dx = c.rawX - sg[0]; val dy = c.rawY - sg[1]; m[c.timestamp] = dx * dx + dy * dy
        }
    }
    Teacher.RTS, Teacher.RTS_EM -> {
        val (xs, ps) = rtsFull(tr)
        tr.obs.indices.associate { i ->
            val (zx, zy) = tr.f.geoToLocal(tr.obs[i].latitude, tr.obs[i].longitude)
            val dx = zx - xs[i][0]; val dy = zy - xs[i][1]
            // EM term: the smoothed position is itself uncertain, residuals alone understate the noise
            val em = if (teacher == Teacher.RTS_EM) ps[i][0][0] + ps[i][1][1] else 0.0
            tr.obs[i].timestamp to dx * dx + dy * dy + em
        }
    }
}

/** Self-training loop: filter with the current R → teacher track → retrain R. Round 0 starts from accuracy². */
fun learnRFrom(train: List<Pair<Truth, List<Observation>>>, q: Double, withNu: Boolean, teacher: Teacher, rounds: Int): RModel {
    var rFn = accuracyR
    repeat(rounds) {
        val rows = ArrayList<DoubleArray>(); val ys = ArrayList<Double>()
        for ((t, o) in train) {
            val ctxs = ArrayList<Pair<RCtx, Long>>()
            val tr = runKfR(o, q, rFn) { ctx, ob -> ctxs += ctx to ob.timestamp }
            val target = teacherTargets(teacher, t, tr)
            for ((ctx, ts) in ctxs) target[ts]?.let { rows += rFeatures(ctx, withNu); ys += ln(it + 0.01) }
        }
        rFn = fitLogR(rows, ys, withNu)
    }
    return rFn
}

/** No-network baseline: classic EM-style calibration R = k·accuracy², k from the RTS teacher. */
fun emCalibrate(train: List<Pair<Truth, List<Observation>>>, q: Double, rounds: Int): RModel {
    var k = 1.0
    repeat(rounds) {
        var num = 0.0; var n = 0
        val kk = k
        for ((t, o) in train) {
            val tr = runKfR(o, q, { c -> kk * accuracyR(c) })
            val target = teacherTargets(Teacher.RTS_EM, t, tr)
            for (ob in o) target[ob.timestamp]?.let {
                val a = ob.accuracy.toDouble().coerceIn(1.0, 50.0); num += it / 2 / (a * a); n++
            }
        }
        k = num / n
    }
    val kFinal = k
    return { c -> kFinal * accuracyR(c) }
}

// ───────────────────────── noise estimator and ZUPT on GFilter (with Doppler) ─────────────────────────

typealias Still = (Meas) -> Boolean

/** Base filter factory: same motion model / sensors, pluggable R model and stillness detector. */
class Base2(val v: Variant, val qa: Double, val qw: Double) {
    val psi = if (v.motion == Motion.CTRV) 3 else -1
    fun make(r: RModel? = null, still: Still? = null, onStep: ((RCtx, Meas) -> Unit)? = null) =
        GFilter(v.motion, v.ukf, qa, qw, v.doppler, v.gyro, r, still, onStep)
    val params get() = if (v.motion == Motion.CV) "q=$qa" else "σa=$qa σω=$qw"
}

/** Process noise tuned on the train split for the 5 s fixed-lag output. */
fun tuneBase(v: Variant, train: List<Route>): Base2 {
    val grid = if (v.motion == Motion.CV) listOf(0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2).map { it to 0.0 }
    else listOf(0.1, 0.2, 0.4, 0.8, 1.5).flatMap { a -> listOf(0.01, 0.02, 0.05, 0.1, 0.2).map { a to it } }
    val best = grid.minBy { (a, w) -> val b = Base2(v, a, w); trainRmse(train.map { (t, ms) -> t to fixedLag(b.make().run(ms), LAG, b.psi) }) }
    return Base2(v, best.first, best.second)
}

/** The noise network self-trained on the offline smoothed track (full RTS + EM term), no ground truth. */
fun learnRSelf(train: List<Route>, base: Base2, rounds: Int = 3): RModel =
    learnRMulti(train.map { it.second to base }, rounds)

/** Self-training over routes that may come from different situations, each with its own base filter. */
fun learnRMulti(data: List<Pair<List<Meas>, Base2>>, rounds: Int = 3): RModel {
    var rFn: RModel? = null
    repeat(rounds) {
        val rows = ArrayList<DoubleArray>(); val ys = ArrayList<Double>()
        for ((ms, base) in data) {
            val ctxs = HashMap<Long, RCtx>()
            val steps = base.make(rFn, null) { c, m -> ctxs[m.obs.timestamp] = c }.run(ms)
            val (xs, ps) = fullSmooth(steps, base.psi)
            for (i in steps.indices) {
                val c = ctxs[steps[i].ts] ?: continue
                val z = toLocal(ms[i].obs.latitude, ms[i].obs.longitude)
                val e2 = (z.x - xs[i][0]) * (z.x - xs[i][0]) + (z.y - xs[i][1]) * (z.y - xs[i][1]) + ps[i][0][0] + ps[i][1][1]
                rows += rFeatures(c, false); ys += ln(e2 + 0.01)
            }
        }
        rFn = fitLogR(rows, ys, false)
    }
    return rFn!!
}

/** No-network competitor: R = k·accuracy², k by the same EM-style teacher. */
fun emCalibG(train: List<Route>, base: Base2, rounds: Int = 3): RModel {
    var k = 1.0
    repeat(rounds) {
        var num = 0.0; var n = 0; val kk = k
        for ((_, ms) in train) {
            val steps = base.make({ c -> kk * accuracyR(c) }).run(ms)
            val (xs, ps) = fullSmooth(steps, base.psi)
            for (i in steps.indices) {
                val z = toLocal(ms[i].obs.latitude, ms[i].obs.longitude)
                val a = ms[i].obs.accuracy.toDouble().coerceIn(1.0, 50.0)
                num += ((z.x - xs[i][0]) * (z.x - xs[i][0]) + (z.y - xs[i][1]) * (z.y - xs[i][1]) + ps[i][0][0] + ps[i][1][1]) / 2 / (a * a); n++
            }
        }
        k = num / n
    }
    val kf = k
    return { c -> kf * accuracyR(c) }
}

/** Mean innovation NLL on [set]: the truth-free acceptance score tried in stage 3. */
fun meanNll(set: List<List<Meas>>, base: Base2, r: RModel?): Double {
    var s = 0.0; var n = 0
    for (ms in set) { val f = base.make(r); f.run(ms); s += f.nll; n += f.nllN }
    return s / n
}

fun stillFeatures(m: Meas) = doubleArrayOf(m.obs.speed.toDouble(), m.accStd, abs(m.gyroZ), m.speedAcc)

/** Stillness classifier for ZUPT; labels are self-made from the offline smoothed speed. */
fun learnStill(train: List<Route>, base: Base2): Still {
    val rows = ArrayList<DoubleArray>(); val ys = ArrayList<Double>()
    for ((_, ms) in train) {
        val steps = base.make().run(ms)
        val (xs, _) = fullSmooth(steps, base.psi)
        for (i in ms.indices) {
            val speed = if (base.v.motion == Motion.CV) hypot(xs[i][2], xs[i][3]) else abs(xs[i][2])
            rows += stillFeatures(ms[i]); ys += if (speed < 0.25) 1.0 else 0.0
        }
    }
    val norm = FeatureNormalizer.fit(rows)
    val net = NeuralNetwork(NetworkConfig(4, listOf(8, 4), 1, OutputActivation.SIGMOID), seed = 11L)
    NeuralNetworkTrainer(net, random = Random(11L)).train(rows.indices.map { TrainingSample(norm.normalize(rows[it]), doubleArrayOf(ys[it])) }, epochs = 30)
    return { m -> net.predict(norm.normalize(stillFeatures(m)))[0] > 0.5 }
}

// ───────────────────────── robust losses ─────────────────────────

enum class Loss { L2, HUBER, CAUCHY }

/**
 * Robust batch smoother. For a linear model, factor-graph optimisation with an M-estimator is solved exactly
 * by RTS inside iteratively reweighted least squares: reweight each GPS fix by its residual, smooth again.
 */
fun robustSmooth(ms: List<Meas>, base: Base2, loss: Loss, iters: Int = 4): Pair<Array<DoubleArray>, Array<Array<DoubleArray>>> {
    val c = 2.45                                    // sqrt(chi2_2 at 95%)
    val w = HashMap<Long, Double>()
    var out: Pair<Array<DoubleArray>, Array<Array<DoubleArray>>>? = null
    repeat(if (loss == Loss.L2) 1 else iters + 1) {
        val snap = HashMap(w)
        val r: RModel = { ctx -> accuracyR(ctx) / (snap[ctx.obs.timestamp] ?: 1.0) }
        val steps = base.make(r).run(ms)
        out = fullSmooth(steps, base.psi)
        val xs = out!!.first
        for (i in ms.indices) {
            val o = ms[i].obs; val z = toLocal(o.latitude, o.longitude)
            val sigma = o.accuracy.toDouble().coerceIn(1.0, 50.0)
            val u = hypot(z.x - xs[i][0], z.y - xs[i][1]) / sigma
            w[o.timestamp] = when (loss) {
                Loss.HUBER -> if (u <= c) 1.0 else c / u
                Loss.CAUCHY -> 1.0 / (1.0 + (u / c) * (u / c))
                Loss.L2 -> 1.0
            }.coerceAtLeast(0.01)
        }
    }
    return out!!
}

/** Classic robust Kalman update online: inflate R when the normalised innovation is too large (soft instead of hard gate). */
val huberOnlineR: RModel = { ctx ->
    val r = accuracyR(ctx)
    val u = ctx.nu / sqrt(ctx.sigmaPos * ctx.sigmaPos / 2 + r)
    r * max(1.0, u / 2.45)
}

/** Noise network self-trained on a chosen teacher (plain RTS or robust IRLS-RTS), EM term included. */
fun learnRTeacher(data: List<List<Meas>>, base: Base2, loss: Loss, rounds: Int = 3): RModel {
    var rFn: RModel? = null
    repeat(rounds) {
        val rows = ArrayList<DoubleArray>(); val ys = ArrayList<Double>()
        for (ms in data) {
            val ctxs = HashMap<Long, RCtx>()
            base.make(rFn, null) { c, m -> ctxs[m.obs.timestamp] = c }.run(ms)
            val (xs, ps) = robustSmooth(ms, base, loss)
            for (i in ms.indices) {
                val c = ctxs[ms[i].obs.timestamp] ?: continue
                val z = toLocal(ms[i].obs.latitude, ms[i].obs.longitude)
                val e2 = (z.x - xs[i][0]) * (z.x - xs[i][0]) + (z.y - xs[i][1]) * (z.y - xs[i][1]) + ps[i][0][0] + ps[i][1][1]
                rows += rFeatures(c, false); ys += ln(e2 + 0.01)
            }
        }
        rFn = fitLogR(rows, ys, false)
    }
    return rFn!!
}
