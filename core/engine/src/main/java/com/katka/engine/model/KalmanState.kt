package com.katka.engine.model

import com.katka.engine.MatrixOps

/**
 * State vector of the two-dimensional constant-velocity Kalman filter.
 *
 * Positions are local metres relative to the filter reference point. Velocities
 * are metres per second. [P] is the 4 by 4 covariance matrix for
 * `[x, y, vx, vy]`.
 *
 * @property x Local X position in metres.
 * @property y Local Y position in metres.
 * @property vx Local X velocity in metres per second.
 * @property vy Local Y velocity in metres per second.
 * @property P State covariance matrix.
 */
data class KalmanState(
    val x: Double,
    val y: Double,
    val vx: Double,
    val vy: Double,
    val P: Array<DoubleArray>
) {
    /** The state as a plain vector for matrix math. */
    fun toVector(): DoubleArray = doubleArrayOf(x, y, vx, vy)

    /** 1-σ horizontal position uncertainty, √(P[0][0] + P[1][1]) in metres. */
    val positionUncertaintyMeters: Double
        get() = Math.sqrt(P[0][0] + P[1][1])

    /** 1-σ velocity uncertainty, √(P[2][2] + P[3][3]) in m/s. */
    val velocityUncertaintyMs: Double
        get() = Math.sqrt(P[2][2] + P[3][3])

    companion object {

        /** State dimension (x, y, vx, vy). */
        const val DIM = 4

        /** Builds an initial state from the first GPS fix with the given position/velocity variances. */
        fun initial(
            x: Double,
            y: Double,
            posVariance: Double = 100.0,
            velVariance: Double = 10.0
        ): KalmanState {
            val P = MatrixOps.zeros(DIM, DIM)
            P[0][0] = posVariance
            P[1][1] = posVariance
            P[2][2] = velVariance
            P[3][3] = velVariance
            return KalmanState(x = x, y = y, vx = 0.0, vy = 0.0, P = P)
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KalmanState) return false
        return x == other.x && y == other.y && vx == other.vx && vy == other.vy &&
                P.contentDeepEquals(other.P)
    }

    override fun hashCode(): Int {
        var result = x.hashCode()
        result = 31 * result + y.hashCode()
        result = 31 * result + vx.hashCode()
        result = 31 * result + vy.hashCode()
        result = 31 * result + P.contentDeepHashCode()
        return result
    }
}
