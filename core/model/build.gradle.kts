plugins {
    id("java-library")
    alias(libs.plugins.kotlin.jvm)
}

// Public data models, pure JVM.
dependencies {
    testImplementation(libs.junit)
}
