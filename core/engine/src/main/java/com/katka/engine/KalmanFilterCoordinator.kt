package com.katka.engine

import com.katka.data.SensorDataSource
import com.katka.engine.coefficient_startegy.ClassicalCoefficientStrategy
import com.katka.engine.coefficient_startegy.CoefficientStrategy
import com.katka.model.AccuracyMetrics
import com.katka.engine.model.FilterMode
import com.katka.engine.model.FilterResult
import com.katka.model.Observation
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*

/**
 * High-level session runner for the Kalman filter.
 *
 * The coordinator starts a [SensorDataSource], collects observations on the
 * supplied dispatcher, feeds them through [filter] with [strategy] and exposes
 * processed [FilterResult] values as a shared [Flow].
 *
 * @param sensorSource Source that provides GPS/IMU observations.
 * @param filter Stateful Kalman filter instance used for the session.
 * @param strategy Strategy that builds the Kalman gain and measurement noise.
 * @param scope Coroutine scope that owns the shared result stream.
 * @param dispatcher Dispatcher used for observation processing.
 * @param filterMode Label stored in diagnostics for the selected filter mode.
 * @param logger Optional logger for session diagnostics.
 */
class KalmanFilterCoordinator(
    private val sensorSource: SensorDataSource,
    val filter: KalmanFilter,
    private val strategy: CoefficientStrategy,
    private val scope: CoroutineScope,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
    private val filterMode: FilterMode = FilterMode.CLASSICAL,
    private val logger: Logger = Logger.NoOp
) {
    /** Routes the engine's Log.x(TAG, …) calls through the injected [Logger]; silent by default. */
    private val Log = object {
        fun i(tag: String, m: String) = logger.i(m)
        fun d(tag: String, m: String) = logger.d(m)
        fun w(tag: String, m: String) = logger.w(m)
        fun e(tag: String, m: String, t: Throwable? = null) = logger.e(m, t)
    }

    private val resultHistory = mutableListOf<FilterResult>()
    private val rawGpsHistory = mutableListOf<Pair<Double, Double>>()

    private var consecutiveGoodFixes = 0

    private val H: Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0, 0.0)
    )

    private var stepCount        = 0
    private var warningCount     = 0
    private var sessionStartMs   = 0L
    private var lastObsTimestamp = 0L

    /**
     * Shared stream of per-step filter results.
     *
     * The stream starts collecting from [sensorSource] while it has subscribers
     * and replays the latest result to new subscribers.
     */
    val results: Flow<FilterResult> = flow {
        sensorSource.observations.collect { obs ->
            val result = processObservation(obs)
            if (result != null) emit(result)
        }
    }
        .flowOn(dispatcher)
        .onEach { result -> resultHistory.add(result) }
        .shareIn(
            scope   = scope,
            started = SharingStarted.WhileSubscribed(stopTimeoutMillis = 5_000L),
            replay  = 1
        )

    /** Resets state and starts the sensor source for a new session. */
    fun start() {
        filter.reset(strategy)
        resultHistory.clear()
        rawGpsHistory.clear()
        stepCount             = 0
        warningCount          = 0
        sessionStartMs        = System.currentTimeMillis()
        lastObsTimestamp      = 0L
        consecutiveGoodFixes  = 0

        Log.i(TAG, buildString {
            appendLine("╔══════════════════════════════════════════════════╗")
            appendLine("║         СЕССИЯ ФИЛЬТРА КАЛМАНА ЗАПУЩЕНА         ║")
            appendLine("╠══════════════════════════════════════════════════╣")
            appendLine("║  Режим    : $filterMode")
            appendLine("║  Диспетчер: $dispatcher")
            appendLine("╚══════════════════════════════════════════════════╝")
        })

        sensorSource.start()
    }

    /** Stops the sensor source and logs a session summary. */
    fun stop() {
        sensorSource.stop()

        val durationSec = (System.currentTimeMillis() - sessionStartMs) / 1000.0

        Log.i(TAG, buildString {
            appendLine("╔══════════════════════════════════════════════════╗")
            appendLine("║         СЕССИЯ ЗАВЕРШЕНА — ИТОГИ                ║")
            appendLine("╠══════════════════════════════════════════════════╣")
            appendLine("║  Шагов обработано : $stepCount")
            appendLine("║  Предупреждений   : $warningCount")
            appendLine("║  Длительность     : ${"%.1f".format(durationSec)} с")
            appendLine("║  Средний интервал : ${
                if (stepCount > 1) "${"%.0f".format(durationSec * 1000 / stepCount)} мс" else "н/д"
            }")
            appendLine("╚══════════════════════════════════════════════════╝")
        })
    }

    /**
     * Computes current-session metrics against the raw GPS track.
     *
     * This is mainly useful for diagnostics and demos where raw GPS is used as
     * the available reference. For a true benchmark, use a separate ground-truth
     * reference with `MetricsEvaluator`.
     */
    fun computeMetrics(): AccuracyMetrics {
        if (resultHistory.size < 2 || rawGpsHistory.size < 2) return AccuracyMetrics.EMPTY
        val n = minOf(resultHistory.size, rawGpsHistory.size)
        val estimated = resultHistory.take(n).map { it.state.x to it.state.y }
        val reference = rawGpsHistory.take(n)
        val errors = estimated.zip(reference).map { (e, r) ->
            val dx = e.first - r.first
            val dy = e.second - r.second
            kotlin.math.sqrt(dx * dx + dy * dy)
        }
        val rmse      = kotlin.math.sqrt(errors.sumOf { it * it } / n)
        val mae       = errors.sumOf { it } / n
        val maxError  = errors.max()
        val stability = if (n < 2) 0.0 else {
            val deltas = estimated.zipWithNext { a, b ->
                val dx = b.first - a.first; val dy = b.second - a.second
                kotlin.math.sqrt(dx * dx + dy * dy)
            }
            val mean = deltas.sum() / deltas.size
            kotlin.math.sqrt(deltas.sumOf { (it - mean) * (it - mean) } / deltas.size)
        }

        Log.i(TAG, buildString {
            appendLine("┌── МЕТРИКИ ТОЧНОСТИ ─────────────────────────────")
            appendLine("│  Точек : $n")
            appendLine("│  RMSE  : ${"%.3f".format(rmse)} м")
            appendLine("│  MAE   : ${"%.3f".format(mae)} м")
            appendLine("│  MAX Δ : ${"%.3f".format(maxError)} м")
            appendLine("│  Jitter: ${"%.3f".format(stability)} м")
            appendLine("└─────────────────────────────────────────────────")
        })

        return AccuracyMetrics(rmse, mae, maxError, lag = 0.0, stability = stability, sampleCount = n)
    }

    /** Snapshot of all filter results produced this session. */
    fun getHistory(): List<FilterResult>               = resultHistory.toList()

    /** Snapshot of all raw GPS positions (local metres) produced this session. */
    fun getRawGpsHistory(): List<Pair<Double, Double>> = rawGpsHistory.toList()

    /** Validates one observation (warm-up, dedup, dt gating) and runs a single filter step. */
    private fun processObservation(obs: Observation): FilterResult? {
        return try {
            if (obs.timestamp > 0L && obs.timestamp == lastObsTimestamp) {
                Log.d(TAG, "⏭ Дубликат obs.timestamp=${obs.timestamp}, пропускаем")
                return null
            }

            if (!filter.isInitialised) {
                if (obs.accuracy > GPS_WARMUP_MAX_ACCURACY_M) {
                    consecutiveGoodFixes = 0
                    Log.d(TAG, "⏳ Warm-up: пропуск фикса acc=${obs.accuracy}м > ${GPS_WARMUP_MAX_ACCURACY_M}м")
                    return null
                }
                consecutiveGoodFixes++
                if (consecutiveGoodFixes < GPS_WARMUP_MIN_GOOD_FIXES) {
                    Log.d(TAG, "⏳ Warm-up: ожидание стабильного GPS ($consecutiveGoodFixes/${GPS_WARMUP_MIN_GOOD_FIXES})")
                    return null
                }
                Log.i(TAG, "✅ GPS стабилен, acc=${obs.accuracy}м — инициализируем фильтр")
            }

            val isFirstObs = !filter.isInitialised

            val gapMs = if (lastObsTimestamp > 0L)
                (obs.timestamp - lastObsTimestamp).coerceAtLeast(0L)
            else 0L

            if (gapMs == 0L && filter.isInitialised) {
                Log.d(TAG, "⏭ dt=0 после инициализации, пропускаем шаг")
                return null
            }

            lastObsTimestamp = obs.timestamp
            stepCount++

            logIncomingObservation(obs, isFirstObs, gapMs)

            val result = filter.process(obs, strategy)

            val (rawX, rawY) = filter.geoToLocal(obs.latitude, obs.longitude)
            rawGpsHistory.add(rawX to rawY)

            if (!isFirstObs && strategy is ClassicalCoefficientStrategy) {
                strategy.updateInnovation(
                    innovation = result.innovation,
                    H          = H,
                    P_pred     = result.predicted.P
                )
            }

            if (stepCount % LOG_EVERY_N_STEPS == 0) {
                logFilterResult(result, rawX, rawY, gapMs)
            }

            checkAnomalies(result, gapMs)

            result
        } catch (e: Exception) {
            Log.e(TAG, "❌ ИСКЛЮЧЕНИЕ на шаге $stepCount: ${e.message}", e)
            null
        }
    }
    
    /** Logs the incoming observation (init banner on the first fix, one line afterwards). */
    private fun logIncomingObservation(obs: Observation, isFirst: Boolean, gapMs: Long) {
        if (isFirst) {
            Log.i(TAG, buildString {
                appendLine("┌── ШАГ 0: ПЕРВЫЙ GPS-ФИКС (инициализация) ──────")
                appendLine("│  Lat       : ${"%.6f".format(obs.latitude)}°")
                appendLine("│  Lon       : ${"%.6f".format(obs.longitude)}°")
                appendLine("│  Accuracy  : ${"%.1f".format(obs.accuracy)} м  ← начальное R")
                appendLine("│  Provider  : ${obs.provider}")
                appendLine("│  HasIMU    : ${obs.hasImu}")
                appendLine("└─────────────────────────────────────────────────")
            })
            return
        }

        Log.d(TAG, buildString {
            append("[#$stepCount]")
            append("  GPS(${obs.provider})")
            append("  acc=${"%.1f".format(obs.accuracy)}м")
            append("  gap=${gapMs}мс")
            append("  ts_age=${System.currentTimeMillis() - obs.timestamp}мс")
            if (obs.hasImu) append("  IMU(ax=${"%.2f".format(obs.ax)},ay=${"%.2f".format(obs.ay)})")
            if (obs.hasSpeed) append("  spd=${"%.2f".format(obs.speed)}м/с")
        })
    }

    /** Logs the full per-step filter readout (state, K, R, innovation, uncertainty). */
    private fun logFilterResult(result: FilterResult, rawX: Double, rawY: Double, gapMs: Long) {
        val K    = result.kalmanGain
        val kx   = K.getOrNull(0)?.getOrNull(0) ?: 0.0
        val ky   = K.getOrNull(1)?.getOrNull(1) ?: 0.0
        val R    = result.measurementNoiseR
        val rxx  = R.getOrNull(0)?.getOrNull(0) ?: 0.0
        val ryy  = R.getOrNull(1)?.getOrNull(1) ?: 0.0
        val innov = result.innovation
        val innovMag = if (innov.size >= 2)
            kotlin.math.sqrt(innov[0] * innov[0] + innov[1] * innov[1]) else 0.0

        val dx = result.state.x - rawX
        val dy = result.state.y - rawY
        val filterOffset = kotlin.math.sqrt(dx * dx + dy * dy)

        Log.i(TAG, buildString {
            appendLine("┌── ШАГ #$stepCount ─────────────────────────────────────")
            appendLine("│  dt             : ${"%.0f".format(result.dt * 1000)} мс")
            appendLine("│")
            appendLine("│  СОСТОЯНИЕ (x, y, vx, vy):")
            appendLine("│    x  = ${"%.3f".format(result.state.x)} м")
            appendLine("│    y  = ${"%.3f".format(result.state.y)} м")
            appendLine("│    vx = ${"%.3f".format(result.state.vx)} м/с")
            appendLine("│    vy = ${"%.3f".format(result.state.vy)} м/с")
            appendLine("│")
            appendLine("│  КОЭФ. КАЛМАНА (K):")
            appendLine("│    K[0,0] = ${"%.5f".format(kx)}  ${gainQuality(kx)}")
            appendLine("│    K[1,1] = ${"%.5f".format(ky)}  ${gainQuality(ky)}")
            appendLine("│")
            appendLine("│  ШУМ ИЗМЕРЕНИЙ (R):")
            appendLine("│    R[0,0] = ${"%.3f".format(rxx)} м²  (σ=${"%.2f".format(kotlin.math.sqrt(rxx))} м)")
            appendLine("│    R[1,1] = ${"%.3f".format(ryy)} м²  (σ=${"%.2f".format(kotlin.math.sqrt(ryy))} м)")
            appendLine("│")
            appendLine("│  ИННОВАЦИЯ  ||y|| = ${"%.3f".format(innovMag)} м  ${innovQuality(innovMag)}")
            appendLine("│    y[0] = ${"%.3f".format(innov.getOrNull(0) ?: 0.0)} м")
            appendLine("│    y[1] = ${"%.3f".format(innov.getOrNull(1) ?: 0.0)} м")
            appendLine("│")
            appendLine("│  НЕОПРЕДЕЛЁННОСТЬ:")
            appendLine("│    σ_pos = ${"%.3f".format(result.state.positionUncertaintyMeters)} м")
            appendLine("│")
            appendLine("│  СГЛАЖИВАНИЕ: GPS→filtered = ${"%.3f".format(filterOffset)} м")
            appendLine("└─────────────────────────────────────────────────")
        })
    }

    /** Warns on suspicious steps (large innovation, divergence, extreme gain, long gaps). */
    private fun checkAnomalies(result: FilterResult, gapMs: Long) {
        val K   = result.kalmanGain
        val kx  = K.getOrNull(0)?.getOrNull(0) ?: 0.0
        val ky  = K.getOrNull(1)?.getOrNull(1) ?: 0.0
        val innov = result.innovation
        val innovMag = if (innov.size >= 2)
            kotlin.math.sqrt(innov[0] * innov[0] + innov[1] * innov[1]) else 0.0

        if (innovMag > WARN_INNOVATION_M) {
            warningCount++
            Log.w(TAG, "⚠️  [#$stepCount] Большая инновация: ${"%.2f".format(innovMag)} м " +
                    "(порог: $WARN_INNOVATION_M м) — возможен GPS-прыжок или потеря сигнала")
        }
        if (result.state.positionUncertaintyMeters > WARN_UNCERTAINTY_M) {
            warningCount++
            Log.w(TAG, "⚠️  [#$stepCount] Высокая неопределённость: " +
                    "${"%.1f".format(result.state.positionUncertaintyMeters)} м — фильтр расходится?")
        }
        if (kx > WARN_KALMAN_GAIN_HIGH || ky > WARN_KALMAN_GAIN_HIGH) {
            warningCount++
            Log.w(TAG, "⚠️  [#$stepCount] Высокий K: kx=${"%.4f".format(kx)}, ky=${"%.4f".format(ky)} " +
                    "— фильтр почти не сглаживает (R >> P?)")
        }
        if (kx < WARN_KALMAN_GAIN_LOW && kx > 0.0) {
            warningCount++
            Log.w(TAG, "⚠️  [#$stepCount] Низкий K: kx=${"%.6f".format(kx)} " +
                    "— фильтр игнорирует GPS (R << P?)")
        }
        if (gapMs > WARN_DT_SPIKE_MS) {
            warningCount++
            Log.w(TAG, "⚠️  [#$stepCount] Большой пропуск между фиксами: ${gapMs} мс " +
                    "— нет GPS-сигнала? Телефон усыплён?")
        }
    }

    /** Maps a gain value to a short quality label for the log. */
    private fun gainQuality(k: Double): String = when {
        k > WARN_KALMAN_GAIN_HIGH -> "⚠ ВЫСОКИЙ"
        k < WARN_KALMAN_GAIN_LOW  -> "⚠ НИЗКИЙ"
        k in 0.3..0.7             -> "✓ норма"
        else                      -> "~ допустимо"
    }

    /** Maps an innovation magnitude to a short quality label for the log. */
    private fun innovQuality(mag: Double): String = when {
        mag < 2.0  -> "✓ хорошо"
        mag < 5.0  -> "~ умеренно"
        mag < 10.0 -> "⚠ высокая"
        else       -> "❌ аномалия"
    }

    companion object {
        private const val TAG = "KalmanLog"
        private const val WARN_INNOVATION_M      = 10.0
        private const val WARN_UNCERTAINTY_M     = 50.0
        private const val WARN_KALMAN_GAIN_HIGH  = 0.95
        private const val WARN_KALMAN_GAIN_LOW   = 0.001
        private const val WARN_DT_SPIKE_MS       = 5000.0
        private const val LOG_EVERY_N_STEPS      = 1
        private const val GPS_WARMUP_MAX_ACCURACY_M = 15f
        private const val GPS_WARMUP_MIN_GOOD_FIXES = 5
    }
}
