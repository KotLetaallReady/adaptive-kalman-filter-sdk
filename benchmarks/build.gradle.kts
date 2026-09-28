import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    alias(libs.plugins.kotlin.jvm)
    application
}

dependencies {
    implementation(project(":core:engine"))
    testImplementation(libs.junit)
}

// Friend of :core:engine, like its tests: uses engine internals without making them public API.
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
