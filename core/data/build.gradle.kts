plugins {
    id("java-library")
    alias(libs.plugins.kotlin.jvm)
}

// Pure Kotlin/JVM module with the sensor-source abstraction
// (Flow<Observation>). Android implementations live outside this module.
dependencies {
    api(project(":core:model"))
    api(libs.kotlinx.coroutines.core)

    testImplementation(libs.junit)
}
