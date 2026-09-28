package com.katka.engine.coefficient_startegy

import com.katka.engine.model.GainResult
import com.katka.model.Observation

/**
 * Computes the Kalman gain for one correction step. Implement it to replace the
 * measurement-noise model without changing [com.katka.engine.KalmanFilter].
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
     * Receives the innovation of every accepted fix right after [computeGain]; fixes rejected by
     * the outlier gate are not reported. Override it to adapt the noise model.
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
