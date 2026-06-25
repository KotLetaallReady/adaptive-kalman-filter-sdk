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
import com.katka.model.AccuracyMetrics
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
    val results: Flow<FilterResult> = coordinator.results

    fun start() = coordinator.start()

    fun stop() = coordinator.stop()

    fun computeMetrics(): AccuracyMetrics = coordinator.computeMetrics()

    fun getHistory(): List<FilterResult> = coordinator.getHistory()

    fun getRawGpsHistory(): List<Pair<Double, Double>> = coordinator.getRawGpsHistory()

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
            filter: KalmanFilter = KalmanFilter(config.processNoiseStd)
        ): AdaptiveKalmanFilterSdk {
            val coordinator = KalmanFilterCoordinator(
                sensorSource = sensorSource,
                filter = filter,
                strategy = strategy,
                scope = scope,
                dispatcher = dispatcher,
                filterMode = FilterMode.CLASSICAL,
                logger = logger
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
                forgettingB = forgettingB
            )
    }
}
