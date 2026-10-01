package com.katka.engine.coefficient_startegy

import com.katka.engine.MatrixOps
import com.katka.engine.model.GainResult
import com.katka.model.Observation
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Measurement-noise strategy with a learned correction of the reported GPS accuracy:
 * `R = accuracy² · exp(clamp(model(features), -1.0, 2.5))`, i.e. between 0.37x and 12.2x the
 * accuracy-based variance. The model sees only past innovations, so it stays causal (0 s delay);
 * it was trained to match the residuals of the robust offline smoother (see [PretrainedResidualModel]).
 *
 * An optional Huber soft gate then inflates `R` further for a single outlying fix.
 *
 * @param minAccuracyM Lower clamp on reported GPS accuracy.
 * @param maxAccuracyM Upper clamp on reported GPS accuracy.
 * @param robustUpdate Whether to apply the Huber soft gate on large innovations.
 * @param huberThreshold Mahalanobis distance threshold for the Huber gate (2.45 is 95% of χ²₂).
 * @param model Noise model, [PretrainedResidualModel.DEFAULT] by default.
 */
class NeuralResidualNoiseStrategy(
    val minAccuracyM: Float = 1.0f,
    val maxAccuracyM: Float = 50.0f,
    val robustUpdate: Boolean = true,
    val huberThreshold: Double = 2.45,
    val model: PretrainedResidualModel = PretrainedResidualModel.DEFAULT
) : CoefficientStrategy {

    private val nuHistory = DoubleArray(HISTORY_CAPACITY)
    private var nuHead = 0
    private var nuCount = 0

    /** Features of the last [computeGain] call, for training and diagnostics. */
    internal val lastFeatures = DoubleArray(PretrainedResidualModel.FEATURES)

    /** Clamped model output of the last [computeGain] call. */
    internal var lastDelta = 0.0
        private set

    override fun computeGain(
        P_pred: Array<DoubleArray>,
        H: Array<DoubleArray>,
        obs: Observation
    ): GainResult = computeGain(P_pred, H, obs, innovation = null)

    override fun computeGain(
        P_pred: Array<DoubleArray>,
        H: Array<DoubleArray>,
        obs: Observation,
        innovation: DoubleArray?
    ): GainResult {
        val clampedAcc = obs.accuracy.coerceIn(minAccuracyM, maxAccuracyM).toDouble()
        val rRaw = (clampedAcc * clampedAcc).coerceAtLeast(1.0)

        fillFeatures(obs, P_pred, clampedAcc)
        val delta = model.predict(lastFeatures)
        lastDelta = if (delta.isFinite()) delta.coerceIn(MIN_DELTA, MAX_DELTA) else 0.0
        var rEff = rRaw * exp(lastDelta)

        val PHt = MatrixOps.mul(P_pred, MatrixOps.transpose(H))
        val HPHt = MatrixOps.mul(H, PHt)

        if (robustUpdate && innovation != null) {
            val s00 = HPHt[0][0] + rEff; val s01 = HPHt[0][1]
            val s10 = HPHt[1][0]; val s11 = HPHt[1][1] + rEff
            val detS = s00 * s11 - s01 * s10
            if (abs(detS) > 1e-12) {
                val y0 = innovation[0]; val y1 = innovation[1]
                val quad = (y0 * (s11 * y0 - s10 * y1) + y1 * (s00 * y1 - s01 * y0)) / detS
                val u = sqrt(max(0.0, quad))
                if (u > huberThreshold) rEff *= u / huberThreshold
            }
        }

        val R = MatrixOps.diagonal(doubleArrayOf(rEff, rEff))
        val K = MatrixOps.mul(PHt, MatrixOps.inverse(MatrixOps.add(HPHt, R)))

        val IKH = MatrixOps.sub(MatrixOps.identity(P_pred.size), MatrixOps.mul(K, H))
        val term1 = MatrixOps.mul(MatrixOps.mul(IKH, P_pred), MatrixOps.transpose(IKH))
        val term2 = MatrixOps.mul(MatrixOps.mul(K, R), MatrixOps.transpose(K))
        return GainResult(K, MatrixOps.symmetrise(MatrixOps.add(term1, term2)), R)
    }

    override fun onInnovation(
        innovation: DoubleArray,
        H: Array<DoubleArray>,
        P_pred: Array<DoubleArray>
    ) {
        val mag = hypot(innovation[0], innovation[1])
        if (!mag.isFinite()) return
        nuHistory[nuHead] = mag
        nuHead = (nuHead + 1) % HISTORY_CAPACITY
        if (nuCount < HISTORY_CAPACITY) nuCount++
    }

    override fun reset() {
        nuHead = 0
        nuCount = 0
        lastDelta = 0.0
    }

    private fun fillFeatures(obs: Observation, P_pred: Array<DoubleArray>, clampedAcc: Double) {
        var mean = 0.0
        var std = 0.0
        if (nuCount > 0) {
            for (i in 0 until nuCount) mean += nuHistory[i]
            mean /= nuCount
            for (i in 0 until nuCount) { val d = nuHistory[i] - mean; std += d * d }
            std = sqrt(std / nuCount)
        }
        lastFeatures[0] = ln(clampedAcc)
        lastFeatures[1] = ln(1.0 + mean)
        lastFeatures[2] = ln(1.0 + std)
        lastFeatures[3] = if (obs.hasSpeed) obs.speed.toDouble() else 0.0
        lastFeatures[4] = ln(sqrt((P_pred[0][0] + P_pred[1][1]).coerceAtLeast(0.01)))
    }

    private companion object {
        const val HISTORY_CAPACITY = 10
        const val MIN_DELTA = -1.0
        const val MAX_DELTA = 2.5
    }
}
