plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.katka.sdk"
    compileSdk {
        version = release(36)
    }

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    api(project(":core:model"))
    api(project(":core:data"))
    api(project(":core:engine"))
    api(project(":core:android"))
    api(libs.kotlinx.coroutines.core)
}
