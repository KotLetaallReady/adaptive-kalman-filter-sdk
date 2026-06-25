plugins {
    id("java-library")
    alias(libs.plugins.kotlin.jvm)
}

// Pure Kotlin/JVM module with SDK domain models. No Android dependencies, so it
// can be imported and unit-tested anywhere.
dependencies {
    testImplementation(libs.junit)
}
