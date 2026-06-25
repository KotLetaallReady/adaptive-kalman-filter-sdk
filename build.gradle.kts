// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.jvm) apply false
}

group = "com.katka"
version = "1.0.0"

subprojects {
    group = rootProject.group
    version = rootProject.version
}
