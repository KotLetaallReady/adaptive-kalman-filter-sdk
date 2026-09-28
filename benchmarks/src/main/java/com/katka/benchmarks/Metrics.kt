package com.katka.benchmarks

import java.io.File
import java.util.Locale
import kotlin.math.sqrt

/** The first 16 s of every route are not scored: every filter is still converging there. */
const val WARMUP_MS = 16_000L

/** RMSE over every scored point of [set]; used for tuning on the train split. */
fun trainRmse(set: List<Pair<Truth, Track>>): Double {
    var se = 0.0; var n = 0
    for ((truth, track) in set) for ((t, p) in track) {
        if (t < WARMUP_MS) continue
        val g = truth.pos[truth.idx.getValue(t)]; val e = (p - g).norm(); se += e * e; n++
    }
    return sqrt(se / n)
}

/** Error statistics of one method over the test routes, all scored on the same timestamps. */
class Stats(val name: String, val lagS: Int) {
    val errs = ArrayList<Double>(); val segSe = DoubleArray(3); val segN = IntArray(3)
    var jit = 0.0; var jn = 0; var nanos = 0L; var steps = 0L
    var lenEst = 0.0; var lenTrue = 0.0
    val trajRmse = ArrayList<Double>()
    fun rmse() = sqrt(errs.sumOf { it * it } / errs.size)
    fun seg(s: Seg) = sqrt(segSe[s.ordinal] / segN[s.ordinal])
    fun p95() = errs.sorted()[(errs.size * 0.95).toInt()]
    fun jitter() = sqrt(jit / jn)
    fun usPerStep() = nanos / 1000.0 / steps

    /** Path length of the estimate against the true path over the same points, in percent; jitter makes it positive. */
    fun lengthErrPct() = 100 * (lenEst / lenTrue - 1)

    fun add(truth: Truth, track: Track, common: List<Long>) {
        var se = 0.0
        for (t in common) {
            val k = truth.idx.getValue(t); val e = (track.getValue(t) - truth.pos[k]).norm()
            errs += e; segSe[truth.seg[k].ordinal] += e * e; segN[truth.seg[k].ordinal]++; se += e * e
        }
        trajRmse += sqrt(se / common.size)
        for (i in 1 until common.size) {
            val a = common[i - 1]; val b = common[i]
            lenEst += (track.getValue(b) - track.getValue(a)).norm()
            lenTrue += (truth.pos[truth.idx.getValue(b)] - truth.pos[truth.idx.getValue(a)]).norm()
        }
        for (i in 1 until common.size - 1) {
            val a = common[i - 1]; val b = common[i]; val c = common[i + 1]
            if (b - a != 1000L || c - b != 1000L) continue
            val pa = track.getValue(a); val pb = track.getValue(b); val pc = track.getValue(c)
            val ga = truth.pos[truth.idx.getValue(a)]; val gb = truth.pos[truth.idx.getValue(b)]; val gc = truth.pos[truth.idx.getValue(c)]
            val ex = (pc.x - 2 * pb.x + pa.x) - (gc.x - 2 * gb.x + ga.x)
            val ey = (pc.y - 2 * pb.y + pa.y) - (gc.y - 2 * gb.y + ga.y)
            jit += ex * ex + ey * ey; jn++
        }
    }
}

inline fun <T> timed(st: Stats, steps: Int, base: Long = 0L, block: () -> T): T {
    val t0 = System.nanoTime(); val r = block()
    st.nanos += System.nanoTime() - t0 + base; st.steps += steps
    return r
}

/** Locale-independent formatting, so tables and CSV always use a decimal point. */
fun fmt(pattern: String, vararg args: Any?): String = String.format(Locale.ROOT, pattern, *args)

/**
 * Metrics of one experiment in long ("tidy") form: one row per value, ready for a pandas / R / Excel pivot.
 * Columns: table, scenario, method, lag_s, metric, value, params.
 */
class Results {
    private class Row(val table: String, val scenario: String, val method: String, val lagS: Int,
                      val metric: String, val value: Double, val params: String)

    private val rows = ArrayList<Row>()

    fun add(table: String, scenario: String, method: String, lagS: Int, metric: String, value: Double, params: String = "") {
        rows += Row(table, scenario, method, lagS, metric, value, params)
    }

    fun addStats(table: String, scenario: String, s: Stats, params: String = "") {
        val m = mutableListOf("rmse" to s.rmse(), "rmse_straight" to s.seg(Seg.STRAIGHT), "rmse_turn" to s.seg(Seg.TURN),
            "rmse_stop" to s.seg(Seg.STOP), "p95" to s.p95(), "jitter" to s.jitter(), "length_err_pct" to s.lengthErrPct())
        if (s.steps > 0) m += "us_per_step" to s.usPerStep()
        for ((k, v) in m) add(table, scenario, s.name, s.lagS, k, v, params)
    }

    fun addAll(other: Results) { rows += other.rows }

    fun isNotEmpty() = rows.isNotEmpty()

    fun writeCsv(file: File) {
        file.parentFile?.mkdirs()
        file.bufferedWriter().use { w ->
            w.write("table,scenario,method,lag_s,metric,value,params\n")
            for (r in rows) {
                w.write(listOf(r.table, r.scenario, r.method, r.lagS.toString(), r.metric, fmt("%.6f", r.value), r.params)
                    .joinToString(",") { csvField(it) })
                w.write("\n")
            }
        }
    }
}

fun csvField(s: String): String = if (s.any { it == ',' || it == '"' || it == '\n' }) "\"" + s.replace("\"", "\"\"") + "\"" else s
