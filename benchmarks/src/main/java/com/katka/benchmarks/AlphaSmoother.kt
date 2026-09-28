package com.katka.benchmarks

import com.katka.engine.neural.NetworkConfig
import com.katka.engine.neural.NeuralNetwork
import com.katka.engine.neural.NeuralNetworkTrainer
import com.katka.engine.neural.TrainingSample
import com.katka.engine.smoothing.FeatureNormalizer
import com.katka.engine.smoothing.NeuralTrajectorySmoother
import com.katka.engine.smoothing.OptimalAlpha
import com.katka.engine.smoothing.SmootherConfig
import com.katka.engine.smoothing.SmootherInput
import com.katka.engine.smoothing.SmootherWindow
import com.katka.engine.smoothing.TurnDetection
import kotlin.math.hypot
import kotlin.random.Random

// The SDK's neural fixed-lag smoother: out = (1 − α)·base + α·SG, α from an MLP.

val CFG_OLD = SmootherConfig(turnDetection = TurnDetection.POINT_DIFF)
val CFG_NEW = SmootherConfig()

enum class Base { KF, RTS }
enum class Labels { SG_RAW, TRUTH }
data class Pipe(val base: Base, val labels: Labels, val weighted: Boolean, val extra: Boolean, val cfg: SmootherConfig)

/** Walks the SDK smoother window and hands (window, centre, base point, SG point) to [fn]. */
internal fun walk(tr: KfTrace, cfg: SmootherConfig, base: Base, rts: Array<DoubleArray?>?,
                  fn: (SmootherWindow, SmootherInput, DoubleArray, DoubleArray) -> Unit) {
    val w = SmootherWindow(cfg)
    for (i in tr.res.indices) {
        w.push(tr.input(i)); if (!w.isFull) continue
        val c = w.centralInput()
        val b = if (base == Base.KF) doubleArrayOf(c.kfX, c.kfY) else rts!![i - cfg.halfWindow]!!
        fn(w, c, b, w.sgKf())
    }
}

internal fun features(w: SmootherWindow, c: SmootherInput, b: DoubleArray, sg: DoubleArray, extra: Boolean): DoubleArray {
    val f = w.rawFeatures()
    return if (!extra) f else f + doubleArrayOf(hypot(sg[0] - b[0], sg[1] - b[1]), hypot(c.kfX - b[0], c.kfY - b[1]))
}

class Model(val net: NeuralNetwork, val norm: FeatureNormalizer, val finalLoss: Double, val meanLabel: Double, val pipe: Pipe)

/** One trainer for every variant; Base.KF + Labels.SG_RAW + no weights is exactly SmoothingTrainingCollector. */
internal fun train(set: List<Triple<Truth, KfTrace, Array<DoubleArray?>?>>, pipe: Pipe): Model {
    val rows = ArrayList<DoubleArray>(); val labels = ArrayList<Double>(); val weights = ArrayList<Double>()
    for ((truth, tr, rts) in set) walk(tr, pipe.cfg, pipe.base, rts) { w, c, b, sg ->
        val target = if (pipe.labels == Labels.TRUTH) tr.toEngine(truth.pos[truth.idx.getValue(c.timestamp)]) else w.sgRaw()
        val phi = if (pipe.labels == Labels.TRUTH) 0.0 else w.turnSuppressionPhiDeg()
        labels += OptimalAlpha.solve(b, sg, target, phi, pipe.cfg)
        rows += features(w, c, b, sg, pipe.extra)
        weights += (sg[0] - b[0]) * (sg[0] - b[0]) + (sg[1] - b[1]) * (sg[1] - b[1])
    }
    val meanW = weights.average()
    val norm = FeatureNormalizer.fit(rows)
    val samples = rows.indices.map {
        TrainingSample(norm.normalize(rows[it]), doubleArrayOf(labels[it]),
            if (pipe.weighted) (weights[it] / meanW).coerceAtMost(10.0) else 1.0)
    }
    val net = NeuralNetwork(NetworkConfig.default().copy(inputSize = rows[0].size), seed = 42L)
    val losses = NeuralNetworkTrainer(net, random = Random(42L)).train(samples, epochs = 60)
    return Model(net, norm, losses.last(), labels.average(), pipe)
}

internal fun apply(tr: KfTrace, rts: Array<DoubleArray?>?, m: Model): Track {
    val out = HashMap<Long, V2>()
    walk(tr, m.pipe.cfg, m.pipe.base, rts) { w, c, b, sg ->
        val a = m.net.predict(m.norm.normalize(features(w, c, b, sg, m.pipe.extra)))[0].coerceIn(0.0, 1.0)
        out[c.timestamp] = tr.toMy(b[0] + a * (sg[0] - b[0]), b[1] + a * (sg[1] - b[1]))
    }
    return out
}

/** The shipped SDK inference path, used for the KF-based rows. */
fun nnRun(tr: KfTrace, m: Model): Track = nnRunAlphas(tr, m).first

/** SDK inference path, also returning every alpha the network produced. */
internal fun nnRunAlphas(tr: KfTrace, m: Model): Pair<Track, List<Double>> {
    val sm = NeuralTrajectorySmoother(m.net, m.norm, m.pipe.cfg)
    val out = HashMap<Long, V2>(); val alphas = ArrayList<Double>()
    for (i in tr.res.indices) sm.push(tr.input(i))?.let { out[it.timestamp] = tr.toMy(it.outX, it.outY); alphas += it.alpha }
    return out to alphas
}

internal fun rtsNnAlphas(tr: KfTrace, rts: Array<DoubleArray?>, m: Model): Pair<Track, List<Double>> {
    val out = HashMap<Long, V2>(); val alphas = ArrayList<Double>()
    walk(tr, m.pipe.cfg, m.pipe.base, rts) { w, c, b, sg ->
        val a = m.net.predict(m.norm.normalize(features(w, c, b, sg, m.pipe.extra)))[0].coerceIn(0.0, 1.0)
        out[c.timestamp] = tr.toMy(b[0] + a * (sg[0] - b[0]), b[1] + a * (sg[1] - b[1])); alphas += a
    }
    return out to alphas
}

/** Savitzky–Golay over the KF points, i.e. the smoother with α = 1. */
internal fun sgTrack(tr: KfTrace, cfg: SmootherConfig = CFG_NEW): Track {
    val out = HashMap<Long, V2>()
    walk(tr, cfg, Base.KF, null) { _, c, _, sg -> out[c.timestamp] = tr.toMy(sg[0], sg[1]) }
    return out
}

fun describe(alphas: List<Double>): String {
    val s = alphas.sorted()
    fun q(p: Double) = s[((s.size - 1) * p).toInt()]
    return fmt("ср.%.2f [10%%:%.2f 50%%:%.2f 90%%:%.2f] α>0.9:%3d%%",
        s.average(), q(0.1), q(0.5), q(0.9), 100 * s.count { it > 0.9 } / s.size)
}

/** Sum of velocity-heading turn angles inside the window (feature #3 of the fixed SDK). */
internal fun turnDeg(w: SmootherWindow) = w.rawFeatures()[2]

/** SG only where the window turns more than [thresholdDeg]; α from [nn] there, or 1 without a network. */
internal fun gatedRun(tr: KfTrace, rts: Array<DoubleArray?>, cfg: SmootherConfig, thresholdDeg: Double, nn: Model?): Track {
    val out = HashMap<Long, V2>()
    walk(tr, cfg, Base.RTS, rts) { w, c, b, sg ->
        val a = when {
            turnDeg(w) <= thresholdDeg -> 0.0
            nn == null -> 1.0
            else -> nn.net.predict(nn.norm.normalize(features(w, c, b, sg, nn.pipe.extra)))[0].coerceIn(0.0, 1.0)
        }
        out[c.timestamp] = tr.toMy(b[0] + a * (sg[0] - b[0]), b[1] + a * (sg[1] - b[1]))
    }
    return out
}
