package com.katka.benchmarks

import com.katka.model.Observation
import java.util.Random
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToLong
import kotlin.math.sin
import kotlin.math.sqrt

// ───────────────────────── geometry ─────────────────────────

data class V2(val x: Double, val y: Double) {
    operator fun minus(o: V2) = V2(x - o.x, y - o.y)
    fun norm() = hypot(x, y)
}

const val REF_LAT = 55.75
const val REF_LON = 37.62
const val RE = 6_371_000.0
val COS_REF = cos(Math.toRadians(REF_LAT))

/** Simulator frame (metres east/north of [REF_LAT], [REF_LON]) to WGS-84. */
fun toGeo(p: V2): Pair<Double, Double> =
    (REF_LAT + Math.toDegrees(p.y / RE)) to (REF_LON + Math.toDegrees(p.x / (RE * COS_REF)))

fun toLocal(lat: Double, lon: Double) =
    V2(RE * Math.toRadians(lon - REF_LON) * COS_REF, RE * Math.toRadians(lat - REF_LAT))

fun wrap(a: Double) = atan2(sin(a), cos(a))
fun Random.u(a: Double, b: Double) = a + (b - a) * nextDouble()
fun Random.g() = nextGaussian()

/** Estimated positions by timestamp, in the simulator frame. */
typealias Track = Map<Long, V2>

// ───────────────────────── scenarios ─────────────────────────

enum class Kind(val cruise: Double) { WALK(1.4), BIKE(5.0) }

/**
 * GNSS error model.
 * - WHITE: white noise, σ = 4 m, honest accuracy.
 * - REALISTIC: Gauss–Markov bias (τ = 30 s) + white noise + multipath bursts of 10–40 m + dropouts.
 * - MISCAL: open sky and urban zones, but the phone reports ~4 m accuracy in both.
 */
enum class Noise { WHITE, REALISTIC, MISCAL }
enum class Seg { STRAIGHT, TURN, STOP }

/** One scenario of HANDOFF §5 with its fixed seed. */
data class Scenario(val kind: Kind, val noise: Noise, val seed: Long) {
    val label get() = "${kind.name}/${noise.name}"
}

val SCENARIOS = listOf(
    Scenario(Kind.WALK, Noise.WHITE, 1L), Scenario(Kind.WALK, Noise.REALISTIC, 2L), Scenario(Kind.BIKE, Noise.REALISTIC, 3L),
    Scenario(Kind.WALK, Noise.MISCAL, 4L), Scenario(Kind.BIKE, Noise.MISCAL, 5L)
)

class Truth(val ts: LongArray, val pos: Array<V2>, val speed: DoubleArray, val seg: Array<Seg>, val heading: DoubleArray) {
    val idx: Map<Long, Int> = ts.withIndex().associate { it.value to it.index }
}

/** Pedestrian or bike route sampled at 1 Hz: straight legs, turns of 30–120° and stops. */
fun genTruth(rng: Random, kind: Kind, durS: Int): Truth {
    val dtS = 0.1
    var x = 0.0; var y = 0.0; var psi = rng.u(0.0, 2 * PI); var v = kind.cruise
    var mode = Seg.STRAIGHT; var left = rng.u(15.0, 60.0)
    var targetV = kind.cruise * rng.u(0.85, 1.15); var yawRate = 0.0
    var t = 0.0; var nextSample = 1.0; var lastTurnEnd = -100.0
    val ts = ArrayList<Long>(); val pos = ArrayList<V2>(); val spd = ArrayList<Double>(); val seg = ArrayList<Seg>(); val hd = ArrayList<Double>()
    while (ts.size < durS) {
        if (left <= 0) {
            when (mode) {
                Seg.STRAIGHT -> if (rng.nextDouble() < 0.25) {
                    mode = Seg.STOP; left = rng.u(10.0, 40.0); targetV = 0.0
                } else {
                    mode = Seg.TURN
                    val ang = Math.toRadians(rng.u(30.0, 120.0)) * if (rng.nextBoolean()) 1 else -1
                    val d = rng.u(2.0, 5.0)
                    yawRate = ang / d; left = d
                    if (kind == Kind.BIKE) targetV = kind.cruise * 0.7
                }
                else -> {
                    if (mode == Seg.TURN) lastTurnEnd = t
                    mode = Seg.STRAIGHT; left = rng.u(15.0, 60.0)
                    targetV = kind.cruise * rng.u(0.85, 1.15); yawRate = 0.0
                }
            }
        }
        val tau = if (mode == Seg.STOP) 1.5 else 2.0
        v += (targetV - v) * dtS / tau
        psi += (if (mode == Seg.TURN) yawRate else rng.g() * 0.02) * dtS
        x += v * cos(psi) * dtS; y += v * sin(psi) * dtS
        t += dtS; left -= dtS
        if (t >= nextSample - 1e-9) {
            ts += (nextSample * 1000).roundToLong(); pos += V2(x, y); spd += v; hd += psi
            seg += when {
                mode == Seg.STOP && v < 0.3 -> Seg.STOP
                mode == Seg.TURN || t - lastTurnEnd < 3.0 -> Seg.TURN
                else -> Seg.STRAIGHT
            }
            nextSample += 1.0
        }
    }
    return Truth(ts.toLongArray(), pos.toTypedArray(), spd.toDoubleArray(), seg.toTypedArray(), hd.toDoubleArray())
}

/** GNSS fixes for [tr]. The true per-axis σ travels in `altitude` as a side channel for oracle rows; the engine ignores it. */
fun genObs(rng: Random, tr: Truth, noise: Noise): List<Observation> {
    val out = ArrayList<Observation>()
    val tau = 30.0; val sgm = 3.0; val sw = 2.0
    val phi = exp(-1.0 / tau); val qn = sqrt(1 - phi * phi) * sgm
    var bx = rng.g() * sgm; var by = rng.g() * sgm
    var mpLeft = 0; var mpX = 0.0; var mpY = 0.0; var dropLeft = 0
    var urban = rng.nextBoolean(); var zoneLeft = rng.u(30.0, 180.0).toInt()
    for (k in tr.ts.indices) {
        val p = tr.pos[k]
        var ex: Double; var ey: Double; var acc: Double; var trueSigma: Double
        if (noise == Noise.WHITE) {
            val s = 4.0
            ex = rng.g() * s; ey = rng.g() * s; acc = s * rng.u(0.8, 1.2); trueSigma = s
        } else if (noise == Noise.MISCAL) {
            // open sky ↔ urban canyon; the phone keeps reporting ~4 m either way
            if (zoneLeft <= 0) { urban = !urban; zoneLeft = rng.u(60.0, 180.0).toInt() }
            zoneLeft--
            val zg = if (urban) 5.0 else 1.0; val zw = if (urban) 6.0 else 1.5
            bx = phi * bx + sqrt(1 - phi * phi) * zg * rng.g(); by = phi * by + sqrt(1 - phi * phi) * zg * rng.g()
            if (urban && mpLeft == 0 && rng.nextDouble() < 0.05) {
                mpLeft = 1 + rng.nextInt(4)
                val m = rng.u(10.0, 40.0); val a = rng.u(0.0, 2 * PI)
                mpX = m * cos(a); mpY = m * sin(a)
            }
            ex = bx + zw * rng.g(); ey = by + zw * rng.g(); acc = 4.0 * rng.u(0.8, 1.25); trueSigma = sqrt(zg * zg + zw * zw)
            if (mpLeft > 0) { ex += mpX; ey += mpY; mpLeft-- }
        } else {
            bx = phi * bx + qn * rng.g(); by = phi * by + qn * rng.g()
            if (mpLeft == 0 && rng.nextDouble() < 0.015) {
                mpLeft = 1 + rng.nextInt(4)
                val m = rng.u(10.0, 40.0); val a = rng.u(0.0, 2 * PI)
                mpX = m * cos(a); mpY = m * sin(a)
            }
            ex = bx + sw * rng.g(); ey = by + sw * rng.g(); acc = sqrt(sgm * sgm + sw * sw) * rng.u(0.7, 1.3); trueSigma = sqrt(sgm * sgm + sw * sw)
            if (mpLeft > 0) { ex += mpX; ey += mpY; acc *= rng.u(1.0, 3.0); mpLeft-- }
            if (dropLeft > 0) { dropLeft--; continue }
            if (k > 20 && rng.nextDouble() < 0.005) { dropLeft = 1 + rng.nextInt(6); continue }
        }
        val (lat, lon) = toGeo(V2(p.x + ex, p.y + ey))
        out += Observation(
            timestamp = tr.ts[k], latitude = lat, longitude = lon, accuracy = acc.toFloat(),
            altitude = trueSigma, speed = max(0.0, tr.speed[k] + 0.2 * rng.g()).toFloat(), bearing = 0f,
            hasSpeed = true, hasBearing = false, ax = 0.0, ay = 0.0, az = 0.0,
            hasImu = false, provider = "sim"
        )
    }
    return out
}

// ───────────────────────── extra phone sensors ─────────────────────────

/**
 * One GNSS epoch plus what else a phone knows at that moment.
 * speedAcc / bearingAccDeg mirror Location.getSpeedAccuracyMetersPerSecond / getBearingAccuracyDegrees (API 26+).
 */
class Meas(
    val obs: Observation, val speedAcc: Double, val bearingAccDeg: Double,
    val gyroZ: Double,      // yaw rate around gravity (rad/s), independent of how the phone is held
    val accStd: Double,     // std of |linear acceleration| over the last second (m/s²)
    val trueStill: Boolean, val trueSigma: Double
)

/** Doppler speed and bearing, gyro and accelerometer; a separate RNG keeps the GNSS position noise unchanged. */
fun genMeas(seed: Long, tr: Truth, obs: List<Observation>, kind: Kind, noise: Noise): List<Meas> {
    val rng = Random(seed)
    val gBias = rng.g() * 0.01
    val gSig = if (kind == Kind.WALK) 0.08 else 0.03       // gait sway is worse than road vibration
    return obs.map { o ->
        val k = tr.idx.getValue(o.timestamp)
        val v = tr.speed[k]
        val urban = noise == Noise.MISCAL && o.altitude > 5.0
        val sAcc = if (urban) 0.4 else 0.2
        val speed = max(0.0, v + sAcc * rng.g())
        val bSig = min(PI, sAcc / max(v, 0.1))
        val psiMeas = tr.heading[k] + bSig * rng.g()
        val bearingDeg = ((90.0 - Math.toDegrees(psiMeas)) % 360.0 + 360.0) % 360.0   // Android: clockwise from north
        val yaw = if (k == 0) 0.0 else tr.heading[k] - tr.heading[k - 1]
        val still = tr.seg[k] == Seg.STOP
        val acc = if (still) 0.03 + abs(0.02 * rng.g()) + (if (rng.nextDouble() < 0.08) rng.u(0.3, 1.5) else 0.0)
        else max(0.05, (if (kind == Kind.WALK) 1.2 else 0.6) * min(1.0, v) + 0.25 * rng.g())
        Meas(
            o.copy(speed = speed.toFloat(), bearing = bearingDeg.toFloat(), hasBearing = v > 0.5),
            sAcc, Math.toDegrees(bSig), yaw + gBias + gSig * rng.g(), acc, still, o.altitude
        )
    }
}

/** Android bearing (degrees clockwise from north) to a math heading (radians counter-clockwise from east). */
fun bearingToPsi(bearingDeg: Float) = wrap(Math.toRadians(90.0 - bearingDeg))

typealias Route = Pair<Truth, List<Meas>>

/** 60 routes × 600 s: the first 20 for tuning and training, the other 40 for testing. */
fun makeData(kind: Kind, noise: Noise, seed: Long): Pair<List<Route>, List<Route>> {
    val rng = Random(seed)
    val all = List(60) { i -> val t = genTruth(rng, kind, 600); val o = genObs(rng, t, noise); t to genMeas(seed * 1000 + i, t, o, kind, noise) }
    return all.take(20) to all.drop(20)
}

fun makeData(s: Scenario) = makeData(s.kind, s.noise, s.seed)

/** Position-only split of the same routes as [makeData]: 20 train, 40 test. */
fun makeObsData(s: Scenario): Pair<List<Pair<Truth, List<Observation>>>, List<Pair<Truth, List<Observation>>>> {
    val rng = Random(s.seed)
    val train = List(20) { val t = genTruth(rng, s.kind, 600); t to genObs(rng, t, s.noise) }
    val test = List(40) { val t = genTruth(rng, s.kind, 600); t to genObs(rng, t, s.noise) }
    return train to test
}
