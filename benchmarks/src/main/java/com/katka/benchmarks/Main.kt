package com.katka.benchmarks

import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.Future
import kotlin.system.exitProcess

/** One experiment: a list of independent jobs (usually one per scenario), each filling its own [Results]. */
private class Experiment(
    val name: String,
    val title: String,
    val header: String = "",
    val jobs: (outDir: File) -> List<(Results) -> String>
)

private fun perScenario(run: (Scenario, Results) -> String): (File) -> List<(Results) -> String> =
    { _ -> SCENARIOS.map { sc -> { r: Results -> run(sc, r) } } }

private fun perScenarioFile(run: (Scenario, File) -> String): (File) -> List<(Results) -> String> =
    { dir -> SCENARIOS.map { sc -> { _: Results -> run(sc, dir) } } }

private val EXPERIMENTS = listOf(
    Experiment("baseline", "§5.1 без Doppler: KF, UKF, RTS и нейросглаживатель α SDK", jobs = perScenario(::baseline)),
    Experiment("turns", "§5.1 ложные повороты: доля окон с подавлением α до и после исправления TurnDetection", jobs = perScenario(::turns)),
    Experiment("window", "§5.1 размер окна L", jobs = perScenario(::window)),
    Experiment("loss", "§5.1 метки и взвешенный loss для α-сети поверх KF", jobs = perScenario(::lossCheck)),
    Experiment("leaderboard", "§5.1 без Doppler: NN-оценка шума и «SG на поворотах» поверх KF и RTS", jobs = perScenario(::leaderboard)),
    Experiment("noise-input", "§5.3 входы сети-оценщика шума: с текущей инновацией и без", jobs = perScenario(::noiseInput)),
    Experiment("noise-teacher", "§5.3 учитель для сети-оценщика шума без истины", jobs = perScenario(::noiseTeacher)),
    Experiment("stage1", "§5.2 этап 1: лучшая основа (фильтр + сглаживание 5 с), позиция / + Doppler / + гироскоп", jobs = perScenario(::stage1)),
    Experiment("stage2", "§5.3 этап 2: механизмы поверх KF-CV + Doppler + RTS", jobs = perScenario { sc, r -> stage2(sc, r) }),
    Experiment("stage3", "§5.4 этап 3: дообучение и приёмка сети-оценщика шума", jobs = { listOf(::stage3) }),
    Experiment("stage4", "§5.5 этап 4: робастные потери и робастный учитель", jobs = perScenario(::stage4)),
    Experiment("gate", "шаг 1: потеря трека гейтом, движок SDK на всех 60 маршрутах (потерян = ≥10 отброшенных подряд)",
        header = GATE_HEADER, jobs = perScenario(::gate)),
    Experiment("summary", "§5.6 лучшая комбинация для каждого режима против SDK как есть", jobs = perScenario(::summary)),
    Experiment("tracks", "CSV: первый тестовый маршрут каждого сценария по точкам, для графиков", jobs = perScenarioFile(::tracks)),
    Experiment("dataset", "CSV: все маршруты с истиной и сенсорами (например, для KalmanNet)", jobs = perScenarioFile(::dataset))
)

/** Everything that reproduces a table; `tracks` and `dataset` only export data. */
private val ALL = EXPERIMENTS.filter { it.name != "tracks" && it.name != "dataset" }
private val EXPORT = EXPERIMENTS.filter { it.name == "tracks" || it.name == "dataset" }

private fun usage() = buildString {
    appendLine("Usage: ./gradlew :benchmarks:run [--args=\"<experiment> ...\"]")
    appendLine("  all     every table (default)")
    appendLine("  export  tracks + dataset")
    for (e in EXPERIMENTS) appendLine(fmt("  %-14s %s", e.name, e.title))
}

fun main(args: Array<String>) {
    val names = args.flatMap { it.split(',', ' ') }.map { it.trim() }.filter { it.isNotEmpty() }.ifEmpty { listOf("all") }
    if (names.any { it == "help" || it == "--help" || it == "-h" }) { print(usage()); return }
    val selected = names.flatMap { n ->
        when (n) {
            "all" -> ALL
            "export" -> EXPORT
            else -> listOf(EXPERIMENTS.find { it.name == n } ?: run { System.err.print("unknown experiment: $n\n" + usage()); exitProcess(2) })
        }
    }.distinct()

    val outDir = File(System.getProperty("benchmarks.out") ?: "build/results")
    outDir.mkdirs()
    val threads = Runtime.getRuntime().availableProcessors()
    val pool = Executors.newFixedThreadPool(threads)
    val t0 = System.nanoTime()
    val futures: List<Pair<Experiment, List<Pair<Future<String>, Results>>>> = selected.map { e ->
        e to e.jobs(outDir).map { job -> val r = Results(); pool.submit(Callable { job(r) }) to r }
    }
    println("benchmarks: ${selected.joinToString(" ") { it.name }} · ${futures.sumOf { it.second.size }} jobs on $threads threads")
    try {
        for ((e, jobs) in futures) {
            val text = buildString {
                appendLine().appendLine("##### ${e.name} — ${e.title}")
                append(e.header)
                for ((f, _) in jobs) append(f.get())
            }
            print(text)
            File(outDir, "${e.name}.txt").writeText(text)
            val merged = Results().apply { jobs.forEach { addAll(it.second) } }
            if (merged.isNotEmpty()) merged.writeCsv(File(outDir, "${e.name}.csv"))
            System.out.flush()
        }
    } finally {
        pool.shutdownNow()
    }
    println()
    println(fmt("results: %s (%.0f s)", outDir.absolutePath, (System.nanoTime() - t0) / 1e9))
}
