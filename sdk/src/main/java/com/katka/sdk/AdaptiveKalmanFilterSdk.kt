package com.katka.sdk

import android.content.Context
import com.katka.android.AndroidLogger
import com.katka.android.AndroidSensorDataSource
import com.katka.android.FileSmootherStore
import com.katka.data.SensorDataSource
import com.katka.engine.KalmanFilter
import com.katka.engine.KalmanFilterCoordinator
import com.katka.engine.Logger
import com.katka.engine.coefficient_startegy.ClassicalCoefficientStrategy
import com.katka.engine.coefficient_startegy.CoefficientStrategy
import com.katka.engine.model.FilterMode
import com.katka.engine.model.FilterResult
import com.katka.engine.neural.SmootherRepository
import com.katka.engine.neural.SmootherStore
import com.katka.engine.smoothing.rts.FixedLagRtsSmoother
import com.katka.engine.smoothing.rts.OfflineTrackSmoother
import com.katka.engine.smoothing.rts.SmoothedResult
import com.katka.model.AccuracyMetrics
import com.katka.model.Observation
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow

/**
 * Ready-to-use facade over the Kalman engine and optional Android adapters.
 *
 * Host applications own Android runtime permission requests. Before calling
 * [start] with [android], request either ACCESS_FINE_LOCATION or
 * ACCESS_COARSE_LOCATION from the user.
 */
class AdaptiveKalmanFilterSdk private constructor(
    val sensorSource: SensorDataSource,
    val filter: KalmanFilter,
    val strategy: CoefficientStrategy,
    val coordinator: KalmanFilterCoordinator,
    val smootherRepository: SmootherRepository?
) {
    /** Real-time, zero-delay filtered state stream (for instantaneous "I am here" map marker). */
    val results: Flow<FilterResult> = coordinator.results

    /**
     * Fixed-lag RTS smoothed stream delayed by [KalmanSdkConfig.rtsLag] samples (for drawing high-accuracy polylines).
     */
    val smoothedResults: Flow<SmoothedResult> = coordinator.smoothedResults

    fun start() = coordinator.start()

    fun stop() = coordinator.stop()

    fun computeMetrics(): AccuracyMetrics = coordinator.computeMetrics()

    fun getHistory(): List<FilterResult> = coordinator.getHistory()

    fun getRawGpsHistory(): List<Pair<Double, Double>> = coordinator.getRawGpsHistory()

    /** Returns all smoothed RTS points produced during the active or completed session. */
    fun getSmoothedHistory(): List<SmoothedResult> = coordinator.getSmoothedHistory()

    /**
     * Smooths an entire recorded track offline using robust Iteratively Reweighted Least Squares (IRLS)
     * with Cauchy loss and backward RTS recursion.
     *
     * @param observations The full recorded GPS track.
     * @param processNoiseStd Process noise parameter (defaults to the filter's setting).
     * @param iterations Number of IRLS reweighting iterations (defaults to 4).
     * @param cauchyScale Scale factor c for Cauchy loss (defaults to 2.45).
     * @param useDoppler Whether to fuse Doppler velocity measurements (defaults to filter's setting).
     * @return List of smoothed points matching [observations].
     */
    fun smoothTrack(
        observations: List<Observation>,
        processNoiseStd: Double = filter.processNoiseStd,
        iterations: Int = 4,
        cauchyScale: Double = 2.45,
        useDoppler: Boolean = filter.useDoppler
    ): List<SmoothedResult> {
        val smoother = OfflineTrackSmoother(
            processNoiseStd = processNoiseStd,
            iterations = iterations,
            cauchyScale = cauchyScale,
            useDoppler = useDoppler
        )
        return smoother.smooth(observations)
    }

    companion object {
        /**
         * Creates a complete Android-backed SDK instance.
         *
         * The host app must provide a lifecycle-aware [scope] and request
         * location permission before [AdaptiveKalmanFilterSdk.start].
         */
        fun android(
            context: Context,
            scope: CoroutineScope,
            config: KalmanSdkConfig = KalmanSdkConfig(),
            dispatcher: CoroutineDispatcher = Dispatchers.Default,
            logger: Logger = AndroidLogger(config.logcatTag)
        ): AdaptiveKalmanFilterSdk {
            val appContext = context.applicationContext
            return create(
                sensorSource = AndroidSensorDataSource(
                    context = appContext,
                    gpsIntervalMs = config.gpsIntervalMs,
                    minDisplacementM = config.minDisplacementM
                ),
                scope = scope,
                config = config,
                dispatcher = dispatcher,
                logger = logger,
                smootherStore = FileSmootherStore(
                    context = appContext,
                    fileName = config.smootherModelFileName
                )
            )
        }

        /**
         * Creates an SDK instance from a caller-provided sensor source.
         *
         * Use this for tests, recorded route playback, server/JVM consumers or
         * host apps that already have their own location pipeline.
         */
        fun create(
            sensorSource: SensorDataSource,
            scope: CoroutineScope,
            config: KalmanSdkConfig = KalmanSdkConfig(),
            dispatcher: CoroutineDispatcher = Dispatchers.Default,
            logger: Logger = Logger.NoOp,
            smootherStore: SmootherStore? = null,
            strategy: CoefficientStrategy = config.createClassicalStrategy(),
            filter: KalmanFilter = KalmanFilter(config.processNoiseStd, config.useWallClockTime, config.useDoppler)
        ): AdaptiveKalmanFilterSdk {
            val rtsSmoother = if (config.enableRtsSmoother) FixedLagRtsSmoother(config.rtsLag) else null
            val coordinator = KalmanFilterCoordinator(
                sensorSource = sensorSource,
                filter = filter,
                strategy = strategy,
                scope = scope,
                dispatcher = dispatcher,
                filterMode = FilterMode.CLASSICAL,
                logger = logger,
                rtsSmoother = rtsSmoother
            )

            return AdaptiveKalmanFilterSdk(
                sensorSource = sensorSource,
                filter = filter,
                strategy = strategy,
                coordinator = coordinator,
                smootherRepository = smootherStore?.let(::SmootherRepository)
            )
        }

        private fun KalmanSdkConfig.createClassicalStrategy(): ClassicalCoefficientStrategy =
            ClassicalCoefficientStrategy(
                minAccuracyM = minAccuracyM,
                maxAccuracyM = maxAccuracyM,
                adaptiveR = adaptiveR,
                adaptiveWindow = adaptiveWindow,
                forgettingB = forgettingB,
                robustUpdate = robustUpdate,
                huberThreshold = huberThreshold
            )
    }
}
