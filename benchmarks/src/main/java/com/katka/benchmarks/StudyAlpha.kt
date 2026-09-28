package com.katka.benchmarks

import com.katka.engine.smoothing.OptimalAlpha
import com.katka.engine.smoothing.SmootherConfig
import kotlin.math.max
import kotlin.math.min

// HANDOFF §5.1: position only, the SDK neural α-smoother against KF, UKF and RTS.

private const val HEADER = "%-34s %4s %6s %6s %6s %6s %6s %6s %7s %7s %7s"
private const val ROW = "%-34s %3dс %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %+6.1f%% %4d/%-2d %7s"

private fun StringBuilder.statsTable(stats: List<Stats>, ref: Stats) {
    appendLine(fmt(HEADER, "метод", "лаг", "RMSE", "прямо", "поворт", "стоп", "p95", "дрожь", "к RTS", "vsRTS", "мкс/шаг"))
    for (s in stats) {
        val wins = s.trajRmse.indices.count { s.trajRmse[it] < ref.trajRmse[it] - 1e-9 }
        appendLine(fmt(ROW, s.name, s.lagS, s.rmse(), s.seg(Seg.STRAIGHT), s.seg(Seg.TURN), s.seg(Seg.STOP), s.p95(), s.jitter(),
            100 * (s.rmse() / ref.rmse() - 1), wins, s.trajRmse.size, if (s.steps > 0) fmt("%.1f", s.usPerStep()) else "—"))
    }
}

private fun Results.addTable(table: String, sc: Scenario, stats: List<Stats>, ref: Stats, params: String) {
    for (s in stats) {
        addStats(table, sc.label, s, params)
        add(table, sc.label, s.name, s.lagS, "rel_to_rts_pct", 100 * (s.rmse() / ref.rmse() - 1), params)
        add(table, sc.label, s.name, s.lagS, "routes_better_than_rts", s.trajRmse.indices.count { s.trajRmse[it] < ref.trajRmse[it] - 1e-9 }.toDouble(), params)
    }
}

/** `baseline`: every method tuned on train; the α-smoother before and after the turn fix, RTS and the oracle α. */
fun baseline(sc: Scenario, out: Results): String {
    val (train, test) = makeObsData(sc)
    val lag = CFG_NEW.halfWindow

    // ── tune every classical method on the train split ──
    val qGrid = listOf(0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0)
    val qKf = qGrid.minBy { q -> trainRmse(train.map { (t, o) -> t to runKf(o, q, false).track() }) }
    val qRts = qGrid.minBy { q -> trainRmse(train.map { (t, o) -> val tr = runKf(o, q, false); t to rtsTrack(tr, rtsEngine(tr, lag)) }) }
    val grid = listOf(0.1, 0.2, 0.4, 0.8, 1.5, 3.0).flatMap { sa -> listOf(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4).map { sa to it } }
    val ukfP = grid.minBy { (sa, sw) -> trainRmse(train.map { (t, o) -> t to Ctrv(sa, sw, true).run(o) }) }

    // ── train every neural variant on the same train split ──
    val kfSet = train.map { (t, o) -> Triple(t, runKf(o, qKf, false), null as Array<DoubleArray?>?) }
    val rtsSet = train.map { (t, o) -> val tr = runKf(o, qRts, false); Triple(t, tr, rtsEngine(tr, lag) as Array<DoubleArray?>?) }
    val mOld = train(kfSet, Pipe(Base.KF, Labels.SG_RAW, false, false, CFG_OLD))
    val mFix = train(kfSet, Pipe(Base.KF, Labels.SG_RAW, false, false, CFG_NEW))
    val mRtsRaw = train(rtsSet, Pipe(Base.RTS, Labels.SG_RAW, false, false, CFG_NEW))
    val mRtsTruth = train(rtsSet, Pipe(Base.RTS, Labels.TRUTH, false, false, CFG_NEW))
    val mRtsW = train(rtsSet, Pipe(Base.RTS, Labels.TRUTH, true, false, CFG_NEW))
    val mRtsWx = train(rtsSet, Pipe(Base.RTS, Labels.TRUTH, true, true, CFG_NEW))

    val names = listOf(
        "UKF-CTRV" to 0, "KF-CV (SDK)" to 0,
        "KF+NN, SDK до исправления" to lag, "KF+NN, SDK после исправл." to lag, "KF+SG α=1 (без сети)" to lag,
        "RTS (без сети)" to lag,
        "RTS+NN, метки SG(raw)" to lag, "RTS+NN, метки от истины" to lag,
        "RTS+NN, истина+вес. loss" to lag, "RTS+NN, истина+вес.+8 призн." to lag,
        "Оракул RTS↔SG (потолок)" to lag
    )
    val stats = names.map { (n, l) -> Stats(n, l) }
    for ((truth, obs) in test) {
        val n = obs.size
        val t0 = System.nanoTime(); val kf = runKf(obs, qKf, false); val kfNs = System.nanoTime() - t0
        stats[1].nanos += kfNs; stats[1].steps += n
        val t1 = System.nanoTime(); val kfR = runKf(obs, qRts, false); val rts = rtsEngine(kfR, lag); val rtsNs = System.nanoTime() - t1
        stats[5].nanos += rtsNs; stats[5].steps += n
        val oracle = HashMap<Long, V2>()
        walk(kfR, CFG_NEW, Base.RTS, rts) { _, c, b, sg ->
            val a = OptimalAlpha.solve(b, sg, kfR.toEngine(truth.pos[truth.idx.getValue(c.timestamp)]), 0.0, CFG_NEW)
            oracle[c.timestamp] = kfR.toMy(b[0] + a * (sg[0] - b[0]), b[1] + a * (sg[1] - b[1]))
        }
        val tracks = listOf(
            timed(stats[0], n) { Ctrv(ukfP.first, ukfP.second, true).run(obs) },
            kf.track(),
            timed(stats[2], n, kfNs) { nnRun(kf, mOld) },
            timed(stats[3], n, kfNs) { nnRun(kf, mFix) },
            timed(stats[4], n, kfNs) { sgTrack(kf) },
            rtsTrack(kfR, rts),
            timed(stats[6], n, rtsNs) { apply(kfR, rts, mRtsRaw) },
            timed(stats[7], n, rtsNs) { apply(kfR, rts, mRtsTruth) },
            timed(stats[8], n, rtsNs) { apply(kfR, rts, mRtsW) },
            timed(stats[9], n, rtsNs) { apply(kfR, rts, mRtsWx) },
            oracle
        )
        val common = tracks.map { it.keys }.reduce { a, b -> a intersect b }.filter { it >= WARMUP_MS }.sorted()
        stats.forEachIndexed { i, s -> s.add(truth, tracks[i], common) }
    }

    val ref = stats[5]
    val params = "KF q=$qKf; RTS q=$qRts; UKF σa=${ukfP.first} σω=${ukfP.second}"
    out.addTable("baseline", sc, stats, ref, params)
    for ((m, name) in listOf(mOld to "KF+NN, SDK до исправления", mFix to "KF+NN, SDK после исправл.", mRtsRaw to "RTS+NN, метки SG(raw)", mRtsTruth to "RTS+NN, метки от истины"))
        out.add("baseline", sc.label, name, lag, "mean_alpha_label", m.meanLabel, params)
    return buildString {
        appendLine().appendLine("=== §5.1 исходная постановка · ${sc.label}  (train 20×600 с, test 40×600 с) ===")
        appendLine("подбор: KF q=$qKf, RTS q=$qRts, UKF σa/σω=$ukfP")
        appendLine("средняя метка α: SDK до=${fmt("%.2f", mOld.meanLabel)}, SDK после=${fmt("%.2f", mFix.meanLabel)}, " +
            "RTS SG(raw)=${fmt("%.2f", mRtsRaw.meanLabel)}, RTS истина=${fmt("%.2f", mRtsTruth.meanLabel)}")
        statsTable(stats, ref)
    }
}

/** `turns`: how often turn suppression fires on straight legs, turns and stops, before and after the fix. */
fun turns(sc: Scenario, out: Results): String {
    val rng = java.util.Random(sc.seed)
    val traces = List(20) { val t = genTruth(rng, sc.kind, 600); t to runKf(genObs(rng, t, sc.noise), 0.2, false) }
    val variants = listOf("до" to CFG_OLD, "после" to CFG_NEW, "после, порог 1 м/с" to CFG_NEW.copy(minTurnSpeedMs = 1.0))
    val sb = StringBuilder()
    for ((name, cfg) in variants) {
        var n = 0; var sum = 0.0
        val supp = IntArray(3); val cnt = IntArray(3)
        for ((t, tr) in traces) walk(tr, cfg, Base.KF, null) { w, c, b, sg ->
            val phi = w.turnSuppressionPhiDeg()
            sum += OptimalAlpha.solve(b, sg, w.sgRaw(), phi, cfg); n++
            val k = t.idx.getValue(c.timestamp)
            val hasTurn = (max(0, k - cfg.halfWindow)..min(t.ts.size - 1, k + cfg.halfWindow)).any { t.seg[it] == Seg.TURN }
            val cls = if (hasTurn) 1 else if (t.seg[k] == Seg.STOP) 2 else 0
            cnt[cls]++; if (phi >= cfg.alphaSuppressionTurnDeg * (1 - cfg.minTurnSuppression)) supp[cls]++
        }
        val pct = IntArray(3) { 100 * supp[it] / max(1, cnt[it]) }
        sb.appendLine(fmt("  %-16s %-20s средняя метка α=%.2f | подавлено окон: прямые %3d%%, с поворотом %3d%%, стоянки %3d%%",
            sc.label, name, sum / n, pct[0], pct[1], pct[2]))
        out.add("turns", sc.label, name, cfg.halfWindow, "mean_alpha_label", sum / n)
        out.add("turns", sc.label, name, cfg.halfWindow, "suppressed_straight_pct", pct[0].toDouble())
        out.add("turns", sc.label, name, cfg.halfWindow, "suppressed_turn_pct", pct[1].toDouble())
        out.add("turns", sc.label, name, cfg.halfWindow, "suppressed_stop_pct", pct[2].toDouble())
    }
    return sb.toString()
}

/** `window`: window length L (delay L/2 s) for SG, the α-smoother and RTS. */
fun window(sc: Scenario, out: Results): String {
    val (train, test) = makeObsData(sc)
    val qGrid = listOf(0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0)
    val qKf = qGrid.minBy { q -> trainRmse(train.map { (t, o) -> t to runKf(o, q, false).track() }) }
    val qRts = qGrid.minBy { q -> trainRmse(train.map { (t, o) -> val tr = runKf(o, q, false); t to rtsTrack(tr, rtsEngine(tr, 5)) }) }

    val trainKf = train.map { (t, o) -> Triple(t, runKf(o, qKf, false), null as Array<DoubleArray?>?) }
    val trainR = train.map { (t, o) -> t to runKf(o, qRts, false) }
    val testKf = test.map { (t, o) -> t to runKf(o, qKf, false) }
    val testR = test.map { (t, o) -> t to runKf(o, qRts, false) }

    val windows = listOf(5, 7, 9, 11, 15, 21, 31)
    val methods = listOf("KF+SG α=1", "KF+NN", "RTS", "RTS+NN")
    // tracks[L][method][route]
    val tracks = HashMap<Int, List<List<Track>>>()
    val alphaNn = HashMap<Int, List<Double>>(); val alphaRts = HashMap<Int, List<Double>>()
    for (L in windows) {
        val cfg = SmootherConfig(windowLength = L); val lag = cfg.halfWindow
        val mNn = train(trainKf, Pipe(Base.KF, Labels.SG_RAW, false, false, cfg))
        val mRts = train(trainR.map { (t, tr) -> Triple(t, tr, rtsEngine(tr, lag) as Array<DoubleArray?>?) },
            Pipe(Base.RTS, Labels.TRUTH, true, false, cfg))
        val sg = ArrayList<Track>(); val nn = ArrayList<Track>(); val r = ArrayList<Track>(); val rn = ArrayList<Track>()
        val aN = ArrayList<Double>(); val aR = ArrayList<Double>()
        for (i in test.indices) {
            val kf = testKf[i].second; val kfR = testR[i].second; val rts = rtsEngine(kfR, lag)
            sg += sgTrack(kf, cfg)
            nnRunAlphas(kf, mNn).let { nn += it.first; aN += it.second }
            r += rtsTrack(kfR, rts)
            rtsNnAlphas(kfR, rts, mRts).let { rn += it.first; aR += it.second }
        }
        tracks[L] = listOf(sg, nn, r, rn); alphaNn[L] = aN; alphaRts[L] = aR
    }

    // Score everything on the same timestamps (those every window length produced).
    val stats = windows.associateWith { L -> methods.map { Stats(it, L / 2) } }
    for (i in test.indices) {
        val common = windows.flatMap { L -> tracks.getValue(L).map { it[i].keys } }
            .reduce { a, b -> a intersect b }.filter { it >= WARMUP_MS }.sorted()
        for (L in windows) for (m in methods.indices) stats.getValue(L)[m].add(test[i].first, tracks.getValue(L)[m][i], common)
    }
    val params = "KF q=$qKf; RTS q=$qRts"
    for (L in windows) for (s in stats.getValue(L)) out.addStats("window", sc.label, s, "L=$L; $params")

    return buildString {
        appendLine().appendLine("=== §5.1 размер окна L · ${sc.label} (KF q=$qKf, RTS q=$qRts) ===")
        appendLine(fmt("%3s %4s | %-24s | %-6s | %-24s | %-6s", "L", "лаг", "KF+SG α=1: всё/прям/пов", "KF+NN", "RTS: всё/прям/пов", "RTS+NN"))
        for (L in windows) {
            val s = stats.getValue(L)
            appendLine(fmt("%3d %3dс | %6.2f %6.2f %6.2f     | %6.2f | %6.2f %6.2f %6.2f     | %6.2f",
                L, L / 2, s[0].rmse(), s[0].seg(Seg.STRAIGHT), s[0].seg(Seg.TURN), s[1].rmse(),
                s[2].rmse(), s[2].seg(Seg.STRAIGHT), s[2].seg(Seg.TURN), s[3].rmse()))
        }
        appendLine("α, которые выдаёт сеть (L=11):")
        appendLine("  KF+NN (SDK, после исправления): ${describe(alphaNn.getValue(11))}")
        appendLine("  RTS+NN:                         ${describe(alphaRts.getValue(11))}")
    }
}

/** `loss`: why the KF-based network settles near α ≈ 0.6 while a constant 1.0 scores better: labels × loss weighting. */
fun lossCheck(sc: Scenario, out: Results): String {
    val (train, test) = makeObsData(sc)
    val cfg = SmootherConfig()
    val q = listOf(0.1, 0.2, 0.35, 0.5, 0.8).minBy { q -> trainRmse(train.map { (t, o) -> t to runKf(o, q, false).track() }) }
    val trainKf = train.map { (t, o) -> Triple(t, runKf(o, q, false), null as Array<DoubleArray?>?) }
    val testKf = test.map { (t, o) -> t to runKf(o, q, false) }
    val sgRmse = trainRmse(testKf.map { (t, tr) -> t to sgTrack(tr, cfg) })
    out.add("loss", sc.label, "KF+SG α=1 (без сети)", cfg.halfWindow, "rmse", sgRmse, "q=$q")
    return buildString {
        appendLine().appendLine("=== §5.1 сеть поверх KF, L=11 · ${sc.label} ===")
        appendLine("  KF+SG α=1 (без сети):             RMSE ${fmt("%.2f", sgRmse)}")
        for ((name, pipe) in listOf(
            "метки SG(raw), обычный loss (SDK)" to Pipe(Base.KF, Labels.SG_RAW, false, false, cfg),
            "метки SG(raw), взвешенный loss   " to Pipe(Base.KF, Labels.SG_RAW, true, false, cfg),
            "метки от истины, обычный loss    " to Pipe(Base.KF, Labels.TRUTH, false, false, cfg),
            "метки от истины, взвешенный loss " to Pipe(Base.KF, Labels.TRUTH, true, false, cfg)
        )) {
            val m = train(trainKf, pipe)
            val runs = testKf.map { (t, tr) -> t to nnRunAlphas(tr, m) }
            val rmse = trainRmse(runs.map { it.first to it.second.first })
            val alphas = runs.flatMap { it.second.second }
            appendLine("  $name: RMSE ${fmt("%.2f", rmse)}, α сети ${describe(alphas)}")
            out.add("loss", sc.label, name.trim(), cfg.halfWindow, "rmse", rmse, "q=$q")
            out.add("loss", sc.label, name.trim(), cfg.halfWindow, "mean_alpha", alphas.average(), "q=$q")
        }
    }
}

/** `leaderboard`: noise network and "SG only on turns" on top of KF and RTS, position only. */
fun leaderboard(sc: Scenario, out: Results): String {
    val (train, test) = makeObsData(sc)
    val cfg = SmootherConfig(); val lag = cfg.halfWindow
    val qGrid = listOf(0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2)

    fun kfQ(rFn: RModel) = qGrid.minBy { q -> trainRmse(train.map { (t, o) -> t to runKfR(o, q, rFn).track() }) }
    fun rtsQ(rFn: RModel) = qGrid.minBy { q ->
        trainRmse(train.map { (t, o) -> val tr = runKfR(o, q, rFn); t to rtsTrack(tr, rtsEngine(tr, lag)) }) }

    val grid = listOf(0.1, 0.2, 0.4, 0.8, 1.5, 3.0).flatMap { sa -> listOf(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4).map { sa to it } }
    val ukfP = grid.minBy { (sa, sw) -> trainRmse(train.map { (t, o) -> t to Ctrv(sa, sw, true).run(o) }) }
    val qKf = kfQ(accuracyR); val qSh = qGrid.minBy { q -> trainRmse(train.map { (t, o) -> t to runKf(o, q, true).track() }) }
    val nnR = learnR(train, qKf, true)
    val qKfNn = kfQ(nnR); val qKfOr = kfQ(oracleR)
    val qRts = rtsQ(accuracyR); val qRtsNn = rtsQ(nnR); val qRtsOr = rtsQ(oracleR)

    // turn models and thresholds, tuned on train, for both RTS bases
    val thrGrid = listOf(15.0, 30.0, 45.0, 60.0, 90.0, 1e9)
    fun turnSetup(rFn: RModel, q: Double): Triple<Double, Model, Double> {
        val set = train.map { (t, o) -> val tr = runKfR(o, q, rFn); Triple(t, tr, rtsEngine(tr, lag) as Array<DoubleArray?>?) }
        val m = train(set, Pipe(Base.RTS, Labels.TRUTH, true, false, cfg))
        val thrRule = thrGrid.minBy { th -> trainRmse(set.map { (t, tr, r) -> t to gatedRun(tr, r!!, cfg, th, null) }) }
        val thrNn = thrGrid.minBy { th -> trainRmse(set.map { (t, tr, r) -> t to gatedRun(tr, r!!, cfg, th, m) }) }
        return Triple(thrRule, m, thrNn)
    }
    val (thrRule, mTurn, thrNn) = turnSetup(accuracyR, qRts)
    val (thrRuleNn, mTurnNn, thrNnNn) = turnSetup(nnR, qRtsNn)

    val names = listOf(
        "UKF-CTRV" to 0, "KF (SDK, R из accuracy)" to 0, "KF + Sage-Husa R (SDK)" to 0,
        "KF + NN-оценка шума" to 0, "KF + идеальный R (потолок)" to 0,
        "RTS" to lag, "RTS + NN-оценка шума" to lag, "RTS + идеальный R (потолок)" to lag,
        "RTS + правило «SG на поворотах»" to lag, "RTS + NN на поворотах" to lag,
        "RTS + NN-шум + правило поворот." to lag, "RTS + NN-шум + NN на поворот." to lag
    )
    val stats = names.map { (n, l) -> Stats(n, l) }
    for ((truth, obs) in test) {
        val kfA = runKfR(obs, qRts, accuracyR); val rA = rtsEngine(kfA, lag)
        val kfN = runKfR(obs, qRtsNn, nnR); val rN = rtsEngine(kfN, lag)
        val kfO = runKfR(obs, qRtsOr, oracleR); val rO = rtsEngine(kfO, lag)
        val tracks = listOf(
            Ctrv(ukfP.first, ukfP.second, true).run(obs),
            runKfR(obs, qKf, accuracyR).track(),
            runKf(obs, qSh, true).track(),
            runKfR(obs, qKfNn, nnR).track(),
            runKfR(obs, qKfOr, oracleR).track(),
            rtsTrack(kfA, rA), rtsTrack(kfN, rN), rtsTrack(kfO, rO),
            gatedRun(kfA, rA, cfg, thrRule, null), gatedRun(kfA, rA, cfg, thrNn, mTurn),
            gatedRun(kfN, rN, cfg, thrRuleNn, null), gatedRun(kfN, rN, cfg, thrNnNn, mTurnNn)
        )
        val common = tracks.map { it.keys }.reduce { a, b -> a intersect b }.filter { it >= WARMUP_MS }.sorted()
        stats.forEachIndexed { i, s -> s.add(truth, tracks[i], common) }
    }

    val ref = stats[5]
    fun thr(x: Double) = if (x > 1e8) "никогда" else "${x.toInt()}°"
    val params = "q: KF=$qKf SH=$qSh KF+NN=$qKfNn KF+ideal=$qKfOr RTS=$qRts RTS+NN=$qRtsNn RTS+ideal=$qRtsOr; UKF=$ukfP; " +
        "turn threshold: rule=${thr(thrRule)} NN=${thr(thrNn)} rule+NNnoise=${thr(thrRuleNn)} NN+NNnoise=${thr(thrNnNn)}"
    out.addTable("leaderboard", sc, stats, ref, params)
    return buildString {
        appendLine().appendLine("=== §5.1 сводная таблица без Doppler · ${sc.label} ===")
        appendLine("подбор q: KF=$qKf SH=$qSh KF+NN=$qKfNn KF+ideal=$qKfOr | RTS=$qRts RTS+NN=$qRtsNn RTS+ideal=$qRtsOr | UKF=$ukfP")
        appendLine("порог поворота: правило=${thr(thrRule)}, NN=${thr(thrNn)}; с NN-шумом: правило=${thr(thrRuleNn)}, NN=${thr(thrNnNn)}")
        statsTable(stats, ref)
    }
}

/** `noise-input`: the noise estimator with and without the current innovation among its inputs. */
fun noiseInput(sc: Scenario, out: Results): String {
    val (train, test) = makeObsData(sc)
    val lag = SmootherConfig().halfWindow
    val qGrid = listOf(0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2)
    fun kfQ(rFn: RModel) = qGrid.minBy { q -> trainRmse(train.map { (t, o) -> t to runKfR(o, q, rFn).track() }) }
    fun rtsQ(rFn: RModel) = qGrid.minBy { q ->
        trainRmse(train.map { (t, o) -> val tr = runKfR(o, q, rFn); t to rtsTrack(tr, rtsEngine(tr, lag)) }) }

    val qKf = kfQ(accuracyR)
    val nnNu = learnR(train, qKf, true); val nnLvl = learnR(train, qKf, false)
    val variants = listOf("R из accuracy (SDK)" to accuracyR, "NN-шум, с текущей инновацией" to nnNu,
        "NN-шум, без текущей инновации" to nnLvl, "R = истинная σ зоны" to oracleR)
    val qs = variants.map { (_, r) -> kfQ(r) to rtsQ(r) }

    val stats = variants.flatMap { (n, _) -> listOf(Stats("KF  + $n", 0), Stats("RTS + $n", lag)) }
    for ((truth, obs) in test) {
        val tracks = variants.indices.flatMap { i ->
            val r = variants[i].second
            val kfR = runKfR(obs, qs[i].second, r)
            listOf(runKfR(obs, qs[i].first, r).track(), rtsTrack(kfR, rtsEngine(kfR, lag)))
        }
        val common = tracks.map { it.keys }.reduce { a, b -> a intersect b }.filter { it >= WARMUP_MS }.sorted()
        stats.forEachIndexed { i, s -> s.add(truth, tracks[i], common) }
    }
    stats.forEachIndexed { i, s -> out.addStats("noise-input", sc.label, s, "q=${if (i % 2 == 0) qs[i / 2].first else qs[i / 2].second}") }
    return buildString {
        appendLine().appendLine("=== §5.3 где сеть оценивает шум (без Doppler) · ${sc.label} ===")
        appendLine(fmt("%-44s %4s %6s %6s %6s %6s %6s", "метод", "лаг", "RMSE", "прямо", "поворт", "стоп", "дрожь"))
        for (i in listOf(0, 2, 4, 6, 1, 3, 5, 7)) {
            val s = stats[i]
            appendLine(fmt("%-44s %3dс %6.2f %6.2f %6.2f %6.2f %6.2f", s.name, s.lagS, s.rmse(),
                s.seg(Seg.STRAIGHT), s.seg(Seg.TURN), s.seg(Seg.STOP), s.jitter()))
        }
    }
}

/** `noise-teacher`: training the noise estimator without ground truth, teacher by teacher (no Doppler). */
fun noiseTeacher(sc: Scenario, out: Results): String {
    val (train, test) = makeObsData(sc)
    val lag = CFG_NEW.halfWindow
    val qGrid = listOf(0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2)
    fun kfQ(rFn: RModel) = qGrid.minBy { q -> trainRmse(train.map { (t, o) -> t to runKfR(o, q, rFn).track() }) }
    fun rtsQ(rFn: RModel) = qGrid.minBy { q ->
        trainRmse(train.map { (t, o) -> val tr = runKfR(o, q, rFn); t to rtsTrack(tr, rtsEngine(tr, lag)) }) }

    val q0 = kfQ(accuracyR)
    val variants = listOf(
        "R из accuracy (SDK сейчас)" to accuracyR,
        "EM-калибровка k·acc² (без сети)" to emCalibrate(train, q0, 3),
        "NN, учитель = твой SG-сглаживатель" to learnRFrom(train, q0, false, Teacher.SG, 3),
        "NN, учитель = RTS по треку" to learnRFrom(train, q0, false, Teacher.RTS, 3),
        "NN, учитель = RTS + EM-поправка" to learnRFrom(train, q0, false, Teacher.RTS_EM, 3),
        "NN, RTS+EM, + текущая инновация" to learnRFrom(train, q0, true, Teacher.RTS_EM, 3),
        "NN, учитель = истина (потолок)" to learnRFrom(train, q0, false, Teacher.TRUTH, 3),
        "NN, истина + текущая инновация" to learnRFrom(train, q0, true, Teacher.TRUTH, 3)
    )
    val qs = variants.map { (_, r) -> kfQ(r) to rtsQ(r) }
    val kf = variants.map { Stats(it.first, 0) }; val rt = variants.map { Stats(it.first, lag) }
    for ((truth, obs) in test) {
        val tracks = variants.indices.map { i ->
            val r = variants[i].second; val tr = runKfR(obs, qs[i].second, r)
            runKfR(obs, qs[i].first, r).track() to rtsTrack(tr, rtsEngine(tr, lag))
        }
        val common = tracks.flatMap { listOf(it.first.keys, it.second.keys) }.reduce { a, b -> a intersect b }.filter { it >= WARMUP_MS }.sorted()
        variants.indices.forEach { i -> kf[i].add(truth, tracks[i].first, common); rt[i].add(truth, tracks[i].second, common) }
    }
    variants.indices.forEach { i ->
        out.addStats("noise-teacher", sc.label, kf[i], "q=${qs[i].first}")
        out.addStats("noise-teacher", sc.label, rt[i], "q=${qs[i].second}")
    }
    return buildString {
        appendLine().appendLine("=== §5.3 обучение оценщика шума без истины (без Doppler) · ${sc.label} ===")
        appendLine(fmt("%-36s | %-18s | %-18s", "как получен R", "KF (0 с): RMSE", "RTS (5 с): RMSE"))
        for (i in variants.indices) {
            appendLine(fmt("%-36s | %6.2f  %+6.1f%%   | %6.2f  %+6.1f%%", variants[i].first,
                kf[i].rmse(), 100 * (kf[i].rmse() / kf[0].rmse() - 1), rt[i].rmse(), 100 * (rt[i].rmse() / rt[0].rmse() - 1)))
        }
    }
}
