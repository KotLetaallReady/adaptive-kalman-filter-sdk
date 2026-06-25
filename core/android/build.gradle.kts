plugins {
    alias(libs.plugins.android.library)
}

// Android adapters for the pure-JVM SDK core: implementations of SensorDataSource,
// SmootherStore and Logger. This is the only core module that depends on Android.
android {
    namespace = "com.katka.android"
    compileSdk {
        version = release(36)
    }
    defaultConfig {
        minSdk = 24
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    api(project(":core:engine"))
    api(project(":core:data"))
    api(project(":core:model"))
    api(libs.kotlinx.coroutines.core)
    implementation(libs.play.services.location)
}
