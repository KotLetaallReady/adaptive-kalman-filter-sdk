package com.katka.engine.smoothing

/** Internal sliding window shared by smoother training and inference. */
internal class SmootherWindow(private val config: SmootherConfig = SmootherConfig()) {

    private val buf = ArrayDeque<SmootherInput>(config.windowLength)

    val isFull: Boolean get() = buf.size >= config.windowLength

    /** Appends a step, evicting the oldest once full. */
    fun push(input: SmootherInput) {
        if (buf.size >= config.windowLength) buf.removeFirst()
        buf.addLast(input)
    }

    fun clear() = buf.clear()

    /** The central element of a full window, i.e. the point being smoothed. */
    fun centralInput(): SmootherInput = buf[config.halfWindow]

    /** Savitzky-Golay estimate over the Kalman points. */
    fun sgKf(): DoubleArray = SavitzkyGolaySmoother.smoothCentre2D(kfPoints())

    /** Savitzky-Golay estimate over the raw GPS points used as pseudo-truth. */
    fun sgRaw(): DoubleArray = SavitzkyGolaySmoother.smoothCentre2D(rawPoints())

    /** Raw, un-normalised feature vector for the current window. */
    fun rawFeatures(): DoubleArray = SmootherFeatures.extract(buf, config)

    /** Turn-suppression angle for the optimal-alpha computation. */
    fun turnSuppressionPhiDeg(): Double = SmootherFeatures.suppressionAngleDeg(buf, config)

    private fun kfPoints(): List<DoubleArray> = buf.map { doubleArrayOf(it.kfX, it.kfY) }
    private fun rawPoints(): List<DoubleArray> = buf.map { doubleArrayOf(it.rawX, it.rawY) }
}
