package com.katka.engine.smoothing

/** Internal sliding window shared by smoother training and inference. */
internal class SmootherWindow(private val config: SmootherConfig = SmootherConfig()) {

    private val buf = ArrayDeque<SmootherInput>(config.windowLength)

    val isFull: Boolean get() = buf.size >= config.windowLength

    fun push(input: SmootherInput) {
        if (buf.size >= config.windowLength) buf.removeFirst()
        buf.addLast(input)
    }

    fun clear() = buf.clear()

    fun centralInput(): SmootherInput = buf[config.halfWindow]

    /** Savitzky-Golay estimate over the Kalman points. */
    fun sgKf(): DoubleArray = SavitzkyGolaySmoother.smoothCentre2D(kfPoints())

    /** Savitzky-Golay estimate over the raw GPS points used as pseudo-truth. */
    fun sgRaw(): DoubleArray = SavitzkyGolaySmoother.smoothCentre2D(rawPoints())

    fun rawFeatures(): DoubleArray = SmootherFeatures.extract(buf, config)

    fun turnSuppressionPhiDeg(): Double = SmootherFeatures.suppressionAngleDeg(buf, config)

    private fun kfPoints(): List<DoubleArray> = buf.map { doubleArrayOf(it.kfX, it.kfY) }
    private fun rawPoints(): List<DoubleArray> = buf.map { doubleArrayOf(it.rawX, it.rawY) }
}
