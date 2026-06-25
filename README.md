# Adaptive Kalman Filter SDK for Android GPS Tracking

Adaptive Kalman Filter SDK is an Android and Kotlin GPS filtering library for location smoothing, GPS trajectory tracking, IMU-assisted prediction, adaptive measurement noise, and optional neural post-smoothing.

Android/Kotlin SDK для адаптивной фильтрации GPS-траекторий фильтром Калмана, сглаживания геолокации с IMU-предсказанием и опциональной нейросетевой постобработкой.

Example Android application that demonstrates the algorithm in practice: [KotLetaallReady/AdaptiveKalmanFilter](https://github.com/KotLetaallReady/AdaptiveKalmanFilter).

Пример Android-приложения, показывающий работу алгоритма библиотеки на практике: [KotLetaallReady/AdaptiveKalmanFilter](https://github.com/KotLetaallReady/AdaptiveKalmanFilter).

Keywords: Android Kalman filter SDK, Kotlin GPS filtering library, adaptive Kalman filter Android, GPS location smoothing, GPS trajectory smoothing, Android sensor fusion, IMU GPS filter, neural trajectory smoother, location tracking SDK.

Ключевые слова: Android SDK фильтр Калмана, Kotlin библиотека GPS-фильтрации, адаптивный фильтр Калмана Android, сглаживание GPS, сглаживание геолокации, сглаживание траектории, sensor fusion Android, IMU GPS filter, нейросетевой сглаживатель траектории.

## English

### What This Repository Is

This repository is an SDK/library project, not a runnable mobile application. It contains no launcher `Activity`, no `Application` class, no Compose UI, no Hilt setup, no app navigation and no CSV export screen. A host Android app imports the SDK, owns its lifecycle and permissions, and passes Android objects such as `Context` and `CoroutineScope` into the library.

The main import target is:

```text
:adaptive-kalman-filter-sdk
```

The source folder for that Gradle module is `sdk/`.

### Features

- Adaptive Kalman filter for 2D GPS trajectory filtering.
- Android GPS and IMU observation source with Google Play Services fallback to `LocationManager`.
- Pure Kotlin/JVM engine for custom sensor sources, recorded routes and tests.
- Adaptive GPS measurement-noise strategy based on reported accuracy and optional innovation statistics.
- Configurable fixed-lag neural trajectory smoother through `SmootherConfig`.
- Optional neural smoother model persistence with stored smoother configuration.
- Metrics for trajectory quality analysis: RMSE, MAE, max error, lag and stability.
- Small Android facade API through `AdaptiveKalmanFilterSdk`.

### Architecture

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

### Installation

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

### Android Quick Start

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

### Configurable Neural Smoother

`SmootherConfig` controls the neural fixed-lag smoother. The defaults preserve the original behaviour: `windowLength = 11`, `turnThresholdDeg = 30.0`, `alphaSuppressionTurnDeg = 60.0`, `minTurnSuppression = 0.05`.

```kotlin
val smootherConfig = SmootherConfig(
    windowLength = 15,
    turnThresholdDeg = 25.0,
    alphaSuppressionTurnDeg = 80.0,
    minTurnSuppression = 0.08
)

val collector = SmoothingTrainingCollector(smootherConfig)

val smoother = NeuralTrajectorySmoother(
    network = loaded.network,
    normalizer = FeatureNormalizer(loaded.featureMean, loaded.featureStd),
    config = loaded.smootherConfig
)
```

Use the same `SmootherConfig` for training and inference. New saved models store this config; legacy `v2` models load with default values.

### Custom Sensor Source

Use a custom `SensorDataSource` when the host app already owns GPS, location replay, simulation or server-side data.

```kotlin
import com.katka.data.ReplaySensorDataSource
import com.katka.sdk.AdaptiveKalmanFilterSdk

val kalmanSdk = AdaptiveKalmanFilterSdk.create(
    sensorSource = ReplaySensorDataSource(observations),
    scope = scope
)
```

### Public API

- `AdaptiveKalmanFilterSdk` - ready-to-use facade for Android and custom sensor sources.
- `KalmanSdkConfig` - GPS, filter and smoother configuration.
- `SmootherConfig` - editable neural smoother parameters, including `windowLength` (`L`) and turn suppression.
- `Observation` - one GPS/IMU measurement consumed by the filter.
- `SensorDataSource` - platform-independent stream of observations.
- `AndroidSensorDataSource` - Android GPS/IMU implementation.
- `KalmanFilter` - lower-level stateful Kalman filter.
- `KalmanFilterCoordinator` - session runner that exposes `Flow<FilterResult>`.
- `ClassicalCoefficientStrategy` - default adaptive measurement-noise strategy.
- `MetricsEvaluator` - trajectory quality metrics.
- `NeuralTrajectorySmoother`, `SmoothingTrainingCollector`, `SmootherRepository` - optional neural smoothing pipeline.

### Diploma Materials

- `Диплом.dock` is the scientific basis for the diploma work.
- `diplom.pptx` is the diploma presentation.

These files are supporting academic materials and are not part of the Gradle SDK build.

### Repository Topics

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

### Requirements

- Android Studio or Gradle with a JDK compatible with the Android Gradle Plugin.
- Android Gradle Plugin 9.x. Android Kotlin support is provided by AGP, so Android library modules intentionally do not apply `org.jetbrains.kotlin.android`.
- Android SDK 36 for building this repository.
- Host app minSdk 24 or newer.
- Physical Android device recommended for real GPS/IMU tracking.

## Русский

### Что Это За Репозиторий

Этот репозиторий является SDK/библиотекой, а не запускаемым мобильным приложением. В нём нет launcher `Activity`, класса `Application`, Compose UI, Hilt-настроек, навигации приложения и экрана экспорта CSV. Хост-приложение подключает SDK, управляет жизненным циклом и разрешениями, а также передаёт в библиотеку Android-объекты вроде `Context` и `CoroutineScope`.

Основной модуль для подключения:

```text
:adaptive-kalman-filter-sdk
```

Исходники этого Gradle-модуля находятся в папке `sdk/`.

### Возможности

- Адаптивный фильтр Калмана для двумерной фильтрации GPS-траектории.
- Android-источник GPS и IMU-измерений с Google Play Services и fallback на `LocationManager`.
- Чистое Kotlin/JVM-ядро для своих источников данных, записанных маршрутов и тестов.
- Адаптивная модель шума GPS-измерений на основе accuracy и опциональной статистики инноваций.
- Настраиваемый fixed-lag нейросетевой сглаживатель через `SmootherConfig`.
- Сохранение модели нейросетевого сглаживателя вместе с конфигурацией сглаживания.
- Метрики качества траектории: RMSE, MAE, максимальная ошибка, лаг и стабильность.
- Компактный Android facade API через `AdaptiveKalmanFilterSdk`.

### Архитектура

```text
adaptive-kalman-filter-sdk
  Android SDK facade. Подключается из хост-приложения.

core:model
  Публичные модели данных: Observation, AccuracyMetrics, ComparisonRow.

core:data
  Платформенно-независимый SensorDataSource и ReplaySensorDataSource.

core:engine
  Чистый Kotlin/JVM фильтр Калмана, стратегии коэффициентов, метрики,
  нейросетевой сглаживатель и контракты сохранения модели.

core:android
  Android-адаптеры: AndroidSensorDataSource, AndroidLogger,
  FileSmootherStore и нужные Android manifest permissions.
```

Только `adaptive-kalman-filter-sdk` и `core:android` зависят от Android. Алгоритмическое ядро можно использовать без Android через `core:engine`, `core:data` и `core:model`.

### Подключение

Рекомендуемый вариант для локального подключения исходников через Gradle composite build:

```kotlin
// settings.gradle.kts в хост-приложении
includeBuild("../adaptive-kalman-filter-lib") {
    dependencySubstitution {
        substitute(module("com.katka:adaptive-kalman-filter-sdk"))
            .using(project(":adaptive-kalman-filter-sdk"))
    }
}
```

```kotlin
// app/build.gradle.kts в хост-приложении
dependencies {
    implementation("com.katka:adaptive-kalman-filter-sdk:1.0.0")
}
```

Также можно подключить модули напрямую:

```kotlin
// settings.gradle.kts в хост-приложении
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

Сборка SDK внутри этого репозитория:

```bash
gradlew.bat :adaptive-kalman-filter-sdk:assembleRelease
```

Запуск JVM-тестов ядра:

```bash
gradlew.bat :core:engine:test
```

### Быстрый Старт На Android

SDK добавляет эти разрешения через manifest модуля `core:android`:

```xml
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

Хост-приложение всё равно должно запросить runtime-разрешение на геолокацию перед вызовом `start()`.

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
        // Используйте отфильтрованную GPS-позицию в приложении.
    }
}

kalmanSdk.start()
// ...
kalmanSdk.stop()
```

### Настраиваемый Нейросетевой Сглаживатель

`SmootherConfig` управляет fixed-lag нейросетевым сглаживателем. Значения по умолчанию сохраняют прежнее поведение: `windowLength = 11`, `turnThresholdDeg = 30.0`, `alphaSuppressionTurnDeg = 60.0`, `minTurnSuppression = 0.05`.

```kotlin
val smootherConfig = SmootherConfig(
    windowLength = 15,
    turnThresholdDeg = 25.0,
    alphaSuppressionTurnDeg = 80.0,
    minTurnSuppression = 0.08
)

val collector = SmoothingTrainingCollector(smootherConfig)

val smoother = NeuralTrajectorySmoother(
    network = loaded.network,
    normalizer = FeatureNormalizer(loaded.featureMean, loaded.featureStd),
    config = loaded.smootherConfig
)
```

Используйте один и тот же `SmootherConfig` при обучении и применении модели. Новые сохранённые модели хранят этот конфиг; старые модели формата `v2` загружаются со значениями по умолчанию.

### Свой Источник Данных

Используйте свой `SensorDataSource`, если хост-приложение уже само получает GPS, проигрывает записанный маршрут, симулирует данные или работает с серверным источником.

```kotlin
import com.katka.data.ReplaySensorDataSource
import com.katka.sdk.AdaptiveKalmanFilterSdk

val kalmanSdk = AdaptiveKalmanFilterSdk.create(
    sensorSource = ReplaySensorDataSource(observations),
    scope = scope
)
```

### Публичный API

- `AdaptiveKalmanFilterSdk` - готовый facade для Android и пользовательских источников данных.
- `KalmanSdkConfig` - конфигурация GPS, фильтра и сглаживателя.
- `SmootherConfig` - редактируемые параметры нейросетевого сглаживателя, включая `windowLength` (`L`) и подавление на поворотах.
- `Observation` - одно GPS/IMU-измерение для фильтра.
- `SensorDataSource` - платформенно-независимый поток наблюдений.
- `AndroidSensorDataSource` - Android-реализация GPS/IMU-источника.
- `KalmanFilter` - низкоуровневый stateful фильтр Калмана.
- `KalmanFilterCoordinator` - runner сессии, который отдаёт `Flow<FilterResult>`.
- `ClassicalCoefficientStrategy` - стандартная адаптивная стратегия шума измерений.
- `MetricsEvaluator` - метрики качества траектории.
- `NeuralTrajectorySmoother`, `SmoothingTrainingCollector`, `SmootherRepository` - нейросетевая pipeline сглаживания.

### Дипломные Материалы

- `Диплом.dock` - научная база дипломной работы.
- `diplom.pptx` - презентация для защиты диплома.

Эти файлы являются сопроводительными академическими материалами и не входят в Gradle-сборку SDK.

### Темы Репозитория

Для поиска на GitHub можно добавить такие topics:

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

### Требования

- Android Studio или Gradle с JDK, совместимым с Android Gradle Plugin.
- Android Gradle Plugin 9.x. Поддержка Kotlin в Android-модулях обеспечивается AGP, поэтому Android library-модули намеренно не применяют `org.jetbrains.kotlin.android`.
- Android SDK 36 для сборки этого репозитория.
- minSdk 24 или выше в хост-приложении.
- Для реального GPS/IMU-трекинга рекомендуется физическое Android-устройство.
