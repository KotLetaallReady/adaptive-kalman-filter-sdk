plugins {
    id("java-library")
    alias(libs.plugins.kotlin.jvm)
}

// Pure JVM engine: Kalman filter, smoother, metrics; Android parts come through interfaces.
dependencies {
    api(project(":core:model"))
    api(project(":core:data"))
    api(libs.kotlinx.coroutines.core)

    testImplementation(libs.junit)
}
