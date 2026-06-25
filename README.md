# Adaptive Kalman Filter SDK for Android GPS Tracking

Adaptive Kalman Filter SDK is an Android and Kotlin GPS filtering library for location smoothing, GPS trajectory tracking, IMU-assisted prediction, adaptive measurement noise, and optional neural post-smoothing.

Keywords: Android Kalman filter SDK, Kotlin GPS filtering library, adaptive Kalman filter Android, GPS location smoothing, GPS trajectory smoothing, Android sensor fusion, IMU GPS filter, neural trajectory smoother, location tracking SDK.

## What This Repository Is

This repository is now an SDK/library project, not a runnable mobile application. It contains no launcher `Activity`, no `Application` class, no Compose UI, no Hilt setup, no app navigation and no CSV export screen. A host Android app imports the SDK, owns its lifecycle and permissions, and passes Android objects such as `Context` and `CoroutineScope` into the library.

The main import target is:

```text
:adaptive-kalman-filter-sdk
```

The source folder for that Gradle module is `sdk/`.

## Features

- Adaptive Kalman filter for 2D GPS trajectory filtering.
- Android GPS and IMU observation source with Google Play Services fallback to `LocationManager`.
- Pure Kotlin/JVM engine for custom sensor sources, recorded routes and tests.
- Adaptive GPS measurement-noise strategy based on reported accuracy and optional innovation statistics.
- Optional neural trajectory smoother with model persistence boundary.
- Metrics for trajectory quality analysis: RMSE, MAE, max error, lag and stability.
- Small Android facade API through `AdaptiveKalmanFilterSdk`.

## Architecture

```text
adaptive-kalman-filter-sdk
  Android SDK facade. Use this from host apps.

core:model
  Public data models: Observation, AccuracyMetrics, ComparisonRow.

core:data
  Platform-independent SensorDataSource and ReplaySensorDataSource.

core:engine
  Pure Kotlin/JVM Kalman filter, coefficient strategies, metrics,
  neural smoother and model persistence contracts.

core:android
  Android adapters: AndroidSensorDataSource, AndroidLogger,
  FileSmootherStore and required Android manifest permissions.
```

Only `adaptive-kalman-filter-sdk` and `core:android` depend on Android. The algorithmic engine is reusable without Android through `core:engine`, `core:data` and `core:model`.

## Installation

Recommended local source import through a Gradle composite build:

```kotlin
// settings.gradle.kts in the host Android app
includeBuild("../adaptive-kalman-filter-lib") {
    dependencySubstitution {
        substitute(module("com.katka:adaptive-kalman-filter-sdk"))
            .using(project(":adaptive-kalman-filter-sdk"))
    }
}
```

```kotlin
// app/build.gradle.kts in the host Android app
dependencies {
    implementation("com.katka:adaptive-kalman-filter-sdk:1.0.0")
}
```

Direct multi-module source import also works:

```kotlin
// settings.gradle.kts in the host Android app
include(
    ":adaptive-kalman-filter-sdk",
    ":core:model",
    ":core:data",
    ":core:engine",
    ":core:android"
)

project(":adaptive-kalman-filter-sdk").projectDir = file("../adaptive-kalman-filter-lib/sdk")
project(":core:model").projectDir = file("../adaptive-kalman-filter-lib/core/model")
project(":core:data").projectDir = file("../adaptive-kalman-filter-lib/core/data")
project(":core:engine").projectDir = file("../adaptive-kalman-filter-lib/core/engine")
project(":core:android").projectDir = file("../adaptive-kalman-filter-lib/core/android")
```

Build the SDK inside this repository:

```bash
gradlew.bat :adaptive-kalman-filter-sdk:assembleRelease
```

Run JVM engine tests:

```bash
gradlew.bat :core:engine:test
```

## Android Quick Start

The SDK contributes these manifest permissions through `core:android`:

```xml
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

The host app must still request runtime location permission before `start()`.

```kotlin
import com.katka.sdk.AdaptiveKalmanFilterSdk
import com.katka.sdk.KalmanSdkConfig
import kotlinx.coroutines.launch

val kalmanSdk = AdaptiveKalmanFilterSdk.android(
    context = applicationContext,
    scope = lifecycleScope,
    config = KalmanSdkConfig(
        gpsIntervalMs = 1_000L,
        adaptiveR = true
    )
)

lifecycleScope.launch {
    kalmanSdk.results.collect { result ->
        val state = result.state
        val (latitude, longitude) = kalmanSdk.filter.localToGeo(state.x, state.y)
        // Use the filtered GPS position in your app.
    }
}

kalmanSdk.start()
// ...
kalmanSdk.stop()
```

## Custom Sensor Source

Use a custom `SensorDataSource` when the host app already owns GPS, location replay, simulation or server-side data.

```kotlin
import com.katka.data.ReplaySensorDataSource
import com.katka.sdk.AdaptiveKalmanFilterSdk

val kalmanSdk = AdaptiveKalmanFilterSdk.create(
    sensorSource = ReplaySensorDataSource(observations),
    scope = scope
)
```

## Public API

- `AdaptiveKalmanFilterSdk` - ready-to-use facade for Android and custom sensor sources.
- `KalmanSdkConfig` - GPS, filter and smoother configuration.
- `Observation` - one GPS/IMU measurement consumed by the filter.
- `SensorDataSource` - platform-independent stream of observations.
- `AndroidSensorDataSource` - Android GPS/IMU implementation.
- `KalmanFilter` - lower-level stateful Kalman filter.
- `KalmanFilterCoordinator` - session runner that exposes `Flow<FilterResult>`.
- `ClassicalCoefficientStrategy` - default adaptive measurement-noise strategy.
- `MetricsEvaluator` - trajectory quality metrics.
- `NeuralTrajectorySmoother`, `SmoothingTrainingCollector`, `SmootherRepository` - optional neural smoothing pipeline.

## Repository Topics

For GitHub search, add these repository topics:

```text
android-sdk
kotlin-library
kalman-filter
adaptive-kalman-filter
gps-tracking
gps-filter
location-smoothing
trajectory-smoothing
sensor-fusion
imu
android-location
```

## Requirements

- Android Studio or Gradle with a JDK compatible with the Android Gradle Plugin.
- Android Gradle Plugin 9.x. Android Kotlin support is provided by AGP, so Android library modules intentionally do not apply `org.jetbrains.kotlin.android`.
- Android SDK 36 for building this repository.
- Host app minSdk 24 or newer.
- Physical Android device recommended for real GPS/IMU tracking.
