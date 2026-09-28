plugins {
    id("java-library")
    alias(libs.plugins.kotlin.jvm)
}

// Sensor-source abstraction (Flow<Observation>), pure JVM.
dependencies {
    api(project(":core:model"))
    api(libs.kotlinx.coroutines.core)

    testImplementation(libs.junit)
}
