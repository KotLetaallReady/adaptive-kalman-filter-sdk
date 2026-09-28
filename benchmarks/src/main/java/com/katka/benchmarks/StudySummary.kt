package com.katka.benchmarks

import java.io.File

// HANDOFF §5.6: the best scheme for each use case, next to the SDK as it ships; plus CSV exports.

private class Contender(val name: String, val lagS: Int, val run: (Route) -> Track)

/** lag_s of a smoother over the whole finished track. */
private const val OFFLINE = -1

/** Every contender of the summary, tuned on the train split of [sc]. */
private fun contenders(sc: Scenario, train: List<Route>): Pair<List<Contender>, String> {
    val trainObs = train.map { (t, ms) -> t to ms.map { it.obs } }
    val qGrid = listOf(0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0)
    val qKf = qGrid.minBy { q -> trainRmse(trainObs.map { (t, o) -> t to runKf(o, q, false).track() }) }
    val grid = listOf(0.1, 0.2, 0.4, 0.8, 1.5, 3.0).flatMap { sa -> listOf(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4).map { sa to it } }
    val ukfP = grid.minBy { (sa, sw) -> trainRmse(trainObs.map { (t, o) -> t to Ctrv(sa, sw, true).run(o) }) }
    val base = tuneBase(KF_DOPPLER, train)
    val nnR = learnRSelf(train, base)
    val list = listOf(
        Contender("SDK как есть: KF, q = 0.5", 0) { (_, ms) -> runKf(ms.map { it.obs }, 0.5, false).track() },
        Contender("SDK: KF, q подобран", 0) { (_, ms) -> runKf(ms.map { it.obs }, qKf, false).track() },
        Contender("UKF-CTRV, только позиция", 0) { (_, ms) -> Ctrv(ukfP.first, ukfP.second, true).run(ms.map { it.obs }) },
        Contender("KF + Doppler", 0) { (_, ms) -> filtTrack(base.make().run(ms)) },
        Contender("KF + Doppler + Хьюбер онлайн", 0) { (_, ms) -> filtTrack(base.make(huberOnlineR).run(ms)) },
        Contender("KF + Doppler + RTS", LAG) { (_, ms) -> fixedLag(base.make().run(ms), LAG, base.psi) },
        Contender("KF + Doppler + RTS + NN-шум", LAG) { (_, ms) -> fixedLag(base.make(nnR).run(ms), LAG, base.psi) },
        Contender("робастный RTS (Коши), весь трек", OFFLINE) { (_, ms) ->
            val (xs, _) = robustSmooth(ms, base, Loss.CAUCHY); ms.indices.associate { ms[it].obs.timestamp to V2(xs[it][0], xs[it][1]) }
        }
    )
    return list to "SDK q=$qKf; UKF σa=${ukfP.first} σω=${ukfP.second}; KF+Doppler ${base.params}"
}

/** `summary` (§5.6): real time (0 s), a live route line (5 s) and a finished track, against the shipped SDK. */
fun summary(sc: Scenario, out: Results): String {
    val (train, test) = makeData(sc)
    val (list, params) = contenders(sc, train)
    val stats = list.map { Stats(it.name, it.lagS) }
    for (route in test) {
        val tracks = list.map { it.run(route) }
        val common = tracks.map { it.keys }.reduce { a, b -> a intersect b }.filter { it >= WARMUP_MS }.sorted()
        stats.forEachIndexed { i, s -> s.add(route.first, tracks[i], common) }
    }
    val ref = stats[0]
    for (s in stats) {
        out.addStats("summary", sc.label, s, params)
        out.add("summary", sc.label, s.name, s.lagS, "rel_to_sdk_pct", 100 * (s.rmse() / ref.rmse() - 1), params)
    }
    return buildString {
        appendLine().appendLine("=== §5.6 лучшая комбинация · ${sc.label} ===")
        appendLine("подбор: $params")
        appendLine(fmt("%-36s %6s %6s %6s %6s %6s %6s %6s %8s", "схема", "лаг", "RMSE", "прямо", "поворт", "стоп", "p95", "дрожь", "к SDK"))
        for (s in stats) {
            val lag = if (s.lagS == OFFLINE) "весь" else "${s.lagS}с"
            appendLine(fmt("%-36s %6s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %+7.1f%%", s.name, lag, s.rmse(),
                s.seg(Seg.STRAIGHT), s.seg(Seg.TURN), s.seg(Seg.STOP), s.p95(), s.jitter(), 100 * (s.rmse() / ref.rmse() - 1)))
        }
    }
}

/** `tracks`: the first test route of [sc], point by point, for trajectory plots. */
fun tracks(sc: Scenario, dir: File): String {
    val (train, test) = makeData(sc)
    val (list, _) = contenders(sc, train)
    val route = test.first()
    val (truth, ms) = route
    val tracks = list.map { it.run(route) }
    val file = File(dir, "tracks/${sc.kind.name}_${sc.noise.name}.csv")
    file.parentFile.mkdirs()
    file.bufferedWriter().use { w ->
        w.write((listOf("t_ms", "seg", "true_x", "true_y", "gps_x", "gps_y", "accuracy") +
            list.flatMap { listOf(csvField("${it.name} x"), csvField("${it.name} y")) }).joinToString(",") + "\n")
        for (m in ms) {
            val t = m.obs.timestamp; val k = truth.idx.getValue(t)
            val z = toLocal(m.obs.latitude, m.obs.longitude)
            val cells = mutableListOf(t.toString(), truth.seg[k].name, fmt("%.3f", truth.pos[k].x), fmt("%.3f", truth.pos[k].y),
                fmt("%.3f", z.x), fmt("%.3f", z.y), fmt("%.2f", m.obs.accuracy))
            for (tr in tracks) {
                val p = tr[t]
                cells += p?.let { fmt("%.3f", it.x) } ?: ""; cells += p?.let { fmt("%.3f", it.y) } ?: ""
            }
            w.write(cells.joinToString(",") + "\n")
        }
    }
    return "${sc.label}: ${file.path}\n"
}

/** `dataset`: every route of [sc] with truth and all simulated sensors, e.g. for training KalmanNet in PyTorch. */
fun dataset(sc: Scenario, dir: File): String {
    val (train, test) = makeData(sc)
    val file = File(dir, "data/${sc.kind.name}_${sc.noise.name}.csv")
    file.parentFile.mkdirs()
    file.bufferedWriter().use { w ->
        w.write("route,split,t_ms,seg,true_x,true_y,true_speed,true_heading_rad,gps_x,gps_y,lat,lon,accuracy," +
            "speed,speed_acc,bearing_deg,has_bearing,bearing_acc_deg,gyro_z,acc_std,true_still,true_sigma\n")
        (train + test).forEachIndexed { r, (truth, ms) ->
            val split = if (r < train.size) "train" else "test"
            for (m in ms) {
                val o = m.obs; val k = truth.idx.getValue(o.timestamp); val z = toLocal(o.latitude, o.longitude)
                w.write(listOf(r.toString(), split, o.timestamp.toString(), truth.seg[k].name,
                    fmt("%.3f", truth.pos[k].x), fmt("%.3f", truth.pos[k].y), fmt("%.3f", truth.speed[k]), fmt("%.5f", truth.heading[k]),
                    fmt("%.3f", z.x), fmt("%.3f", z.y), fmt("%.8f", o.latitude), fmt("%.8f", o.longitude), fmt("%.3f", o.accuracy),
                    fmt("%.3f", o.speed), fmt("%.3f", m.speedAcc), fmt("%.2f", o.bearing), o.hasBearing.toString(), fmt("%.3f", m.bearingAccDeg),
                    fmt("%.5f", m.gyroZ), fmt("%.4f", m.accStd), m.trueStill.toString(), fmt("%.3f", m.trueSigma)).joinToString(","))
                w.write("\n")
            }
        }
    }
    return "${sc.label}: ${file.path}\n"
}
