import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    alias(libs.plugins.kotlin.jvm)
    application
}

// Research benchmarks for the engine: a synthetic GPS/phone-sensor simulator,
// competitor filters (EKF/UKF, RTS, robust smoothers) and the tables from
// kalman-research/HANDOFF.md. Not published.
dependencies {
    implementation(project(":core:engine"))
    testImplementation(libs.junit)
}

// The benchmarks drive engine internals (MatrixOps, SmootherWindow, OptimalAlpha)
// the same way the engine's own tests do, without making them public API.
val engine = project(":core:engine")
tasks.withType<KotlinCompile>().configureEach {
    friendPaths.from(
        engine.layout.buildDirectory.dir("classes/kotlin/main"),
        engine.layout.buildDirectory.dir("libs")
    )
}

application {
    mainClass.set("com.katka.benchmarks.MainKt")
    applicationDefaultJvmArgs = listOf("-Dfile.encoding=UTF-8", "-Dsun.stdout.encoding=UTF-8", "-Dstdout.encoding=UTF-8")
}

tasks.named<JavaExec>("run") {
    systemProperty("benchmarks.out", layout.buildDirectory.dir("results").get().asFile.path)
}
