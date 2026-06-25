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

    /** Clears internal adaptive state before a new tracking session starts. */
    fun reset() {}
}
