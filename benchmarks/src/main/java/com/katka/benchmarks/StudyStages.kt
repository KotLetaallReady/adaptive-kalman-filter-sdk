package com.katka.benchmarks

import kotlin.math.sqrt

// Phone sensors (Doppler, gyro, accelerometer), then mechanisms on top of KF-CV + Doppler + RTS.

/** `stage1`: which Kalman-family filter + smoother gives the best route, with and without Doppler and gyro. */
fun stage1(sc: Scenario, out: Results): String {
    val (train, test) = makeData(sc)
    val cvGrid = listOf(0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2).map { it to 0.0 }
    val ctGrid = listOf(0.1, 0.2, 0.4, 0.8, 1.5).flatMap { a -> listOf(0.01, 0.02, 0.05, 0.1, 0.2).map { a to it } }
    fun filterOf(v: Variant, p: Pair<Double, Double>) = GFilter(v.motion, v.ukf, p.first, p.second, v.doppler, v.gyro)
    val psiOf = { v: Variant -> if (v.motion == Motion.CTRV) 3 else -1 }

    val tuned = STAGE1.map { v ->
        (if (v.motion == Motion.CV) cvGrid else ctGrid).minBy { p ->
            trainRmse(train.map { (t, ms) -> t to fixedLag(filterOf(v, p).run(ms), LAG, psiOf(v)) })
        }
    }
    val filt = STAGE1.map { Stats(it.name, 0) }; val smo = STAGE1.map { Stats(it.name, LAG) }
    val nanos = LongArray(STAGE1.size)
    for ((truth, ms) in test) {
        val runs = STAGE1.indices.map { i ->
            val t0 = System.nanoTime(); val steps = filterOf(STAGE1[i], tuned[i]).run(ms); val sm = fixedLag(steps, LAG, psiOf(STAGE1[i]))
            nanos[i] += System.nanoTime() - t0
            filtTrack(steps) to sm
        }
        val common = runs.flatMap { listOf(it.first.keys, it.second.keys) }.reduce { a, b -> a intersect b }.filter { it >= WARMUP_MS }.sorted()
        STAGE1.indices.forEach { i -> filt[i].add(truth, runs[i].first, common); smo[i].add(truth, runs[i].second, common) }
    }
    val steps = test.sumOf { it.second.size }
    val best = smo.indices.minBy { smo[it].rmse() }
    fun params(i: Int) = if (STAGE1[i].motion == Motion.CV) "q=${tuned[i].first}" else "σa=${tuned[i].first} σω=${tuned[i].second}"
    for (i in STAGE1.indices) {
        out.addStats("stage1", sc.label, filt[i], params(i))
        out.addStats("stage1", sc.label, smo[i], params(i))
        out.add("stage1", sc.label, STAGE1[i].name, LAG, "us_per_step", nanos[i] / 1000.0 / steps, params(i))
    }
    // sanity check: the benchmark's KF-CV must match the SDK engine KF + RTS on position-only input
    val eng = trainRmse(test.map { (t, ms) -> val tr = runKf(ms.map { it.obs }, tuned[0].first, false); t to rtsTrack(tr, rtsEngine(tr, LAG)) })
    val mine = trainRmse(test.map { (t, ms) -> t to fixedLag(filterOf(STAGE1[0], tuned[0]).run(ms), LAG, -1) })
    out.add("stage1", sc.label, "SDK engine KF + RTS (parity check)", LAG, "rmse_all_points", eng, params(0))
    out.add("stage1", sc.label, "KF-CV (parity check)", LAG, "rmse_all_points", mine, params(0))

    return buildString {
        appendLine().appendLine("=== этап 1 · ${sc.label} ===")
        appendLine(fmt("%-28s | %-7s | %-38s | %-6s | %s", "фильтр → сглаживатель", "фильтр", "сглаж. 5 с: RMSE / прямо / поворот / стоп", "дрожь", "мкс/шаг  параметры"))
        for (i in STAGE1.indices) {
            val s = smo[i]
            appendLine(fmt("%-28s | %6.2f  | %6.2f  %6.2f  %6.2f  %6.2f  %+6.1f%% | %6.2f | %6.1f   %s%s",
                STAGE1[i].name, filt[i].rmse(), s.rmse(), s.seg(Seg.STRAIGHT), s.seg(Seg.TURN), s.seg(Seg.STOP),
                100 * (s.rmse() / smo[0].rmse() - 1), s.jitter(), nanos[i] / 1000.0 / steps, params(i),
                if (i == best) "   ← лучший" else ""))
        }
        appendLine(fmt("проверка: движок SDK (KF + RTS, q=%.2f) = %.2f м против KF-CV бенчмарка %.2f м (на всех точках)", tuned[0].first, eng, mine))
    }
}

/** `stage2`: ZUPT (rules, classifier, oracle), EM calibration and the self-trained noise network on top of [v]. */
fun stage2(sc: Scenario, out: Results, v: Variant = KF_DOPPLER): String {
    val (train, test) = makeData(sc)
    val base = tuneBase(v, train)

    val ruleDoppler: Still = { m -> m.obs.speed < 0.3 }
    val ruleAcc: Still = { m -> m.accStd < 0.25 }
    val ruleBoth: Still = { m -> m.obs.speed < 0.3 && m.accStd < 0.25 }
    val nnStill = learnStill(train, base)
    val oracleStill: Still = { m -> m.trueStill }
    val nnR = learnRSelf(train, base)
    val emR = emCalibG(train, base)

    // pick the better of rule/NN classifier on train for the combined row
    val stillCands = listOf(ruleDoppler, ruleAcc, ruleBoth, nnStill)
    val bestStill = stillCands.minBy { s -> trainRmse(train.map { (t, ms) -> t to fixedLag(base.make(null, s).run(ms), LAG, base.psi) }) }
    val bestStillName = listOf("Doppler", "аксел.", "Doppler+аксел.", "NN")[stillCands.indexOf(bestStill)]

    val rows = listOf<Pair<String, Pair<RModel?, Still?>>>(
        "основа (R из accuracy)" to (null to null),
        "ZUPT: правило Doppler < 0.3 м/с" to (null to ruleDoppler),
        "ZUPT: правило аксел. < 0.25" to (null to ruleAcc),
        "ZUPT: правило Doppler И аксел." to (null to ruleBoth),
        "ZUPT: NN-классификатор" to (null to nnStill),
        "ZUPT: идеальный (истина)" to (null to oracleStill),
        "R: EM-калибровка k·acc² (без сети)" to (emR to null),
        "R: NN-оценка шума (самообуч.)" to (nnR to null),
        "NN-шум + ZUPT($bestStillName)" to (nnR to bestStill)
    )
    val filt = rows.map { Stats(it.first, 0) }; val smo = rows.map { Stats(it.first, LAG) }
    for ((truth, ms) in test) {
        val runs = rows.map { (_, rs) -> val st = base.make(rs.first, rs.second).run(ms); filtTrack(st) to fixedLag(st, LAG, base.psi) }
        val common = runs.flatMap { listOf(it.first.keys, it.second.keys) }.reduce { a, b -> a intersect b }.filter { it >= WARMUP_MS }.sorted()
        rows.indices.forEach { i -> filt[i].add(truth, runs[i].first, common); smo[i].add(truth, runs[i].second, common) }
    }
    fun quality(s: Still): Pair<Int, Int> {
        var tp = 0; var fp = 0; var fn = 0
        for ((_, ms) in test) for (m in ms) { val p = s(m); if (p && m.trueStill) tp++ else if (p) fp++ else if (m.trueStill) fn++ }
        return 100 * tp / maxOf(1, tp + fp) to 100 * tp / maxOf(1, tp + fn)
    }
    fun qStr(s: Still) = quality(s).let { "точн. ${it.first}% / полнота ${it.second}%" }
    val params = "основа ${v.name}, ${base.params}"
    for (i in rows.indices) {
        out.addStats("stage2", sc.label, filt[i], params)
        out.addStats("stage2", sc.label, smo[i], params)
        out.add("stage2", sc.label, filt[i].name, 0, "rel_to_base_pct", 100 * (filt[i].rmse() / filt[0].rmse() - 1), params)
        out.add("stage2", sc.label, smo[i].name, LAG, "rel_to_base_pct", 100 * (smo[i].rmse() / smo[0].rmse() - 1), params)
    }
    for ((name, s) in listOf("Doppler" to ruleDoppler, "аксел." to ruleAcc, "Doppler+аксел." to ruleBoth, "NN" to nnStill)) {
        val (precision, recall) = quality(s)
        out.add("stage2", sc.label, "классификатор стоянки: $name", 0, "precision_pct", precision.toDouble(), params)
        out.add("stage2", sc.label, "классификатор стоянки: $name", 0, "recall_pct", recall.toDouble(), params)
    }
    return buildString {
        appendLine().appendLine("=== этап 2 · ${sc.label} · основа: ${v.name} (${base.params}) ===")
        appendLine("классификаторы стоянки: Doppler ${qStr(ruleDoppler)} | аксел. ${qStr(ruleAcc)} | оба ${qStr(ruleBoth)} | NN ${qStr(nnStill)}")
        appendLine(fmt("%-36s | %-14s | %-36s | %s", "механизм", "фильтр (0 с)", "сглаж. 5 с: RMSE / прямо / поворот / стоп", "дрожь"))
        for (i in rows.indices) {
            val f = filt[i]; val s = smo[i]
            appendLine(fmt("%-36s | %5.2f %+6.1f%% | %5.2f %+6.1f%%  %5.2f  %5.2f  %5.2f | %5.2f",
                rows[i].first, f.rmse(), 100 * (f.rmse() / filt[0].rmse() - 1), s.rmse(), 100 * (s.rmse() / smo[0].rmse() - 1),
                s.seg(Seg.STRAIGHT), s.seg(Seg.TURN), s.seg(Seg.STOP), s.jitter()))
        }
    }
}

/** `stage3`: how the noise network should be (re)trained and whether a truth-free check can accept it. */
fun stage3(out: Results): String {
    val v = KF_DOPPLER
    val data = SCENARIOS.map { makeData(it) }
    val bases = data.map { tuneBase(v, it.first) }
    // 14 routes to train on, 6 held out for the truth-free acceptance test
    val fit = data.map { it.first.take(14) }; val valid = data.map { it.first.drop(14).map { p -> p.second } }
    val mixed = learnRMulti(SCENARIOS.indices.flatMap { i -> fit[i].take(3).map { it.second to bases[i] } })
    val walkOnly = learnRMulti(fit[1].map { it.second to bases[1] })

    val columns = listOf("accuracy", "EM k·acc²", "NN: свой", "NN: смесь", "NN: 1 сценар.", "страж (NLL)")
    val sb = StringBuilder()
    sb.appendLine().appendLine("=== этап 3 · дообучение оценщика шума на основе KF-CV + Doppler (сглаж. 5 с, RMSE и изменение к R из accuracy) ===")
    sb.appendLine(fmt("%-20s | %-13s | %-13s | %-13s | %-13s | %-13s | %-13s | %s", "сценарий", *columns.toTypedArray(), "выбор стража"))
    for (i in SCENARIOS.indices) {
        val base = bases[i]; val test = data[i].second
        val own = learnRMulti(fit[i].map { it.second to base })
        val em = emCalibG(fit[i], base)
        val cands = listOf<Pair<String, RModel?>>("accuracy" to null, "EM" to em, "NN свой" to own, "NN смесь" to mixed)
        val nlls = cands.map { meanNll(valid[i], base, it.second) }
        val pick = nlls.indices.minBy { nlls[it] }
        val models = listOf<RModel?>(null, em, own, mixed, walkOnly, cands[pick].second)
        val rm = models.map { r -> trainRmse(test.map { (t, ms) -> t to fixedLag(base.make(r).run(ms), LAG, base.psi) }) }
        val sc = SCENARIOS[i]
        columns.indices.forEach { c ->
            out.add("stage3", sc.label, columns[c], LAG, "rmse", rm[c], base.params)
            out.add("stage3", sc.label, columns[c], LAG, "rel_to_accuracy_pct", 100 * (rm[c] / rm[0] - 1), base.params)
        }
        cands.indices.forEach { c -> out.add("stage3", sc.label, "NLL: ${cands[c].first}", LAG, "nll", nlls[c], base.params) }
        sb.appendLine(fmt("%-20s | %s | %s", sc.label,
            rm.joinToString(" | ") { fmt("%5.2f %+6.1f%%", it, 100 * (it / rm[0] - 1)) },
            cands[pick].first + "  (NLL: " + nlls.joinToString(" ") { fmt("%.3f", it) } + ")"))
    }
    return sb.toString()
}

/** `stage4`: robust losses offline (IRLS RTS, Huber / Cauchy), online Huber update, robust teacher. */
fun stage4(sc: Scenario, out: Results): String {
    val (train, test) = makeData(sc)
    val base = tuneBase(KF_DOPPLER, train)
    fun full(loss: Loss) = trainRmse(test.map { (t, ms) ->
        val (xs, _) = robustSmooth(ms, base, loss); t to ms.indices.associate { ms[it].obs.timestamp to V2(xs[it][0], xs[it][1]) }
    })
    fun lagged(r: RModel?) = trainRmse(test.map { (t, ms) -> t to fixedLag(base.make(r).run(ms), LAG, base.psi) })
    fun causal(r: RModel?) = trainRmse(test.map { (t, ms) -> t to filtTrack(base.make(r).run(ms)) })

    val fL2 = full(Loss.L2); val fH = full(Loss.HUBER); val fC = full(Loss.CAUCHY)
    val lagBase = lagged(null); val lagHub = lagged(huberOnlineR)
    val cauBase = causal(null); val cauHub = causal(huberOnlineR)
    val nnPlain = learnRTeacher(train.map { it.second }, base, Loss.L2)
    val nnRobust = learnRTeacher(train.map { it.second }, base, Loss.CAUCHY)
    val lagNnP = lagged(nnPlain); val lagNnR = lagged(nnRobust)
    fun pct(a: Double, b: Double) = fmt("%+.1f%%", 100 * (a / b - 1))

    val offline = -1   // lag_s of a smoother over the whole finished track
    for ((name, lag, value) in listOf(
        Triple("RTS, весь трек", offline, fL2), Triple("робастный RTS Хьюбер, весь трек", offline, fH),
        Triple("робастный RTS Коши, весь трек", offline, fC),
        Triple("KF + Doppler", 0, cauBase), Triple("KF + Doppler + Хьюбер онлайн", 0, cauHub),
        Triple("KF + Doppler + RTS", LAG, lagBase), Triple("KF + Doppler + Хьюбер + RTS", LAG, lagHub),
        Triple("RTS + NN-шум, учитель RTS", LAG, lagNnP), Triple("RTS + NN-шум, учитель Коши", LAG, lagNnR)
    )) out.add("stage4", sc.label, name, lag, "rmse", value, base.params)

    return fmt(
        "%-16s | учитель (офлайн, весь трек): RTS %.2f · Хьюбер %.2f (%s) · Коши %.2f (%s) | онлайн Хьюбер-KF: 0 с %.2f→%.2f (%s), 5 с %.2f→%.2f (%s) | NN-шум от учителя: RTS %.2f (%s) · Коши %.2f (%s)\n",
        sc.label, fL2, fH, pct(fH, fL2), fC, pct(fC, fL2),
        cauBase, cauHub, pct(cauHub, cauBase), lagBase, lagHub, pct(lagHub, lagBase),
        lagNnP, pct(lagNnP, lagBase), lagNnR, pct(lagNnR, lagBase)
    )
}

/**
 * `gate`: the SDK engine KF on all 60 routes of a scenario.
 * "Lost" = the gate rejected at least 10 fixes in a row somewhere on the route.
 */
fun gate(sc: Scenario, out: Results): String {
    val (train, test) = makeData(sc)
    val routes = (train + test).map { (t, ms) -> t to ms.map { it.obs } }
    fun gateStats(q: Double, adaptive: Boolean): Triple<Double, Int, Int> {
        var se = 0.0; var n = 0; var lost = 0; var worst = 0
        for ((truth, obs) in routes) {
            val tr = runKf(obs, q, adaptive)
            var streak = 0; var maxStreak = 0
            for (i in 1 until tr.res.size) {
                val gated = tr.res[i].kalmanGain.all { row -> row.all { it == 0.0 } }
                streak = if (gated) streak + 1 else 0
                if (streak > maxStreak) maxStreak = streak
            }
            if (maxStreak >= 10) lost++
            if (maxStreak > worst) worst = maxStreak
            for ((t, p) in tr.track()) {
                if (t < WARMUP_MS) continue
                val e = (p - truth.pos[truth.idx.getValue(t)]).norm(); se += e * e; n++
            }
        }
        return Triple(sqrt(se / n), lost, worst)
    }
    val a = gateStats(0.2, false); val b = gateStats(0.5, false); val c = gateStats(0.5, true)
    for ((name, s) in listOf("SDK KF, q=0.2" to a, "SDK KF, q=0.5 (по умолчанию)" to b, "SDK KF + Sage–Husa, q=0.5" to c)) {
        out.add("gate", sc.label, name, 0, "rmse_all_points", s.first)
        out.add("gate", sc.label, name, 0, "lost_routes", s.second.toDouble(), "of ${routes.size}")
        out.add("gate", sc.label, name, 0, "max_rejected_in_a_row", s.third.toDouble())
    }
    return fmt("%-18s | %7.2f %6d/%-2d %13d | %7.2f %6d/%-2d %13d | %7.2f\n",
        sc.label, a.first, a.second, routes.size, a.third, b.first, b.second, routes.size, b.third, c.first)
}

const val GATE_HEADER = "                   | q = 0.2                            | q = 0.5 (по умолчанию)             | Sage–Husa, q = 0.5\n" +
    "сценарий           |    RMSE  потеряно  макс. подряд |    RMSE  потеряно  макс. подряд |    RMSE\n"
