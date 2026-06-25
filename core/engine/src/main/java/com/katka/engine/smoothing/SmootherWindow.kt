package com.katka.engine.smoothing

/** Internal sliding window shared by smoother training and inference. */
internal class SmootherWindow(private val length: Int = SmootherFeatures.L) {

    private val buf = ArrayDeque<SmootherInput>(length)

    val isFull: Boolean get() = buf.size >= length

    /** Appends a step, evicting the oldest once full. */
    fun push(input: SmootherInput) {
        if (buf.size >= length) buf.removeFirst()
        buf.addLast(input)
    }

    fun clear() = buf.clear()

    /** The central element of a full window (the point being smoothed). */
    fun centralInput(): SmootherInput = buf[(length - 1) / 2]

    /** Savitzky-Golay estimate over the Kalman points. */
    fun sgKf(): DoubleArray = SavitzkyGolaySmoother.smoothCentre2D(kfPoints())

    /** Savitzky-Golay estimate over the raw GPS points (pseudo-truth). */
    fun sgRaw(): DoubleArray = SavitzkyGolaySmoother.smoothCentre2D(rawPoints())

    /** Raw (un-normalised) 6-feature vector for the current window. */
    fun rawFeatures(): DoubleArray = SmootherFeatures.extract(buf)

    /** Turn-suppression angle for the optimal-alpha computation. */
    fun turnSuppressionPhiDeg(): Double = SmootherFeatures.suppressionAngleDeg(buf)

    private fun kfPoints(): List<DoubleArray> = buf.map { doubleArrayOf(it.kfX, it.kfY) }
    private fun rawPoints(): List<DoubleArray> = buf.map { doubleArrayOf(it.rawX, it.rawY) }
}
