plugins {
    id("java-library")
    alias(libs.plugins.kotlin.jvm)
}

// Pure Kotlin/JVM module: Kalman filter, neural trajectory smoother, metrics
// and math. Android specifics are injected through interfaces and implemented
// in :core:android or host applications.
dependencies {
    api(project(":core:model"))
    api(project(":core:data"))
    api(libs.kotlinx.coroutines.core)

    testImplementation(libs.junit)
}
