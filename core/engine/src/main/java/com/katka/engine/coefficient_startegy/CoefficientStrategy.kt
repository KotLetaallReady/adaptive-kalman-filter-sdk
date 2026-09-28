package com.katka.engine.coefficient_startegy

import com.katka.engine.model.GainResult
import com.katka.model.Observation

/**
 * Strategy that computes the Kalman gain for one correction step.
 *
 * Implement this interface to replace the default measurement-noise model or
 * to experiment with adaptive and learned variants while keeping the main
 * [com.katka.engine.KalmanFilter] unchanged.
 */
interface CoefficientStrategy {

    /**
     * Builds the gain, posterior covariance and measurement noise for a step.
     *
     * @param P_pred Predicted state covariance before measurement correction.
     * @param H Measurement matrix that maps state to observed position.
     * @param obs Observation being processed.
     */
    fun computeGain(
        P_pred: Array<DoubleArray>,
        H: Array<DoubleArray>,
        obs: Observation
    ): GainResult

    /**
     * Receives the innovation of an accepted fix after [computeGain].
     *
     * [com.katka.engine.KalmanFilter.process] calls this once per accepted
     * measurement update. Fixes rejected by the outlier gate are not reported.
     * Override it to adapt the noise model from innovation statistics.
     *
     * @param innovation Measurement residual `z - Hx` against the prediction.
     * @param H Measurement matrix used for the update.
     * @param P_pred Predicted state covariance before the update.
     */
    fun onInnovation(
        innovation: DoubleArray,
        H: Array<DoubleArray>,
        P_pred: Array<DoubleArray>
    ) {}

    /** Clears internal adaptive state before a new tracking session starts. */
    fun reset() {}
}
