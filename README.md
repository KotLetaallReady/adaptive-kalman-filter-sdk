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

benchmarks
  Research benchmarks, not published: a GPS and phone-sensor simulator,
  EKF/UKF and RTS competitors, result tables and CSV exports.
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

`SmootherConfig` controls the neural fixed-lag smoother. Defaults: `windowLength = 11`, `turnThresholdDeg = 30.0`, `alphaSuppressionTurnDeg = 60.0`, `minTurnSuppression = 0.05`, `turnDetection = TurnDetection.VELOCITY`, `minTurnSpeedMs = 0.5`.

Turns are measured from the Kalman velocity heading; samples slower than `minTurnSpeedMs` are skipped. The earlier `TurnDetection.POINT_DIFF` mode (bearing between consecutive filtered points) reported turns on most walking windows, because at walking speed a one-second step is comparable to the filter jitter, and that pushed almost every training label to zero.

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

Use the same `SmootherConfig` for training and inference. New saved models (`v4`) store this config. Legacy `v3` models keep their stored values and `v2` models load with defaults; both load with `TurnDetection.POINT_DIFF`, so their features stay the same. Retrain them to get the fixed turn detection.

### Custom Sensor Source

Use a custom `SensorDataSource` when the host app already owns GPS, location replay, simulation or server-side data.

```kotlin
import com.katka.data.ReplaySensorDataSource
import com.katka.sdk.AdaptiveKalmanFilterSdk
import com.katka.sdk.KalmanSdkConfig

val kalmanSdk = AdaptiveKalmanFilterSdk.create(
    sensorSource = ReplaySensorDataSource(observations),
    scope = scope,
    config = KalmanSdkConfig(useWallClockTime = false)
)
```

For a recorded route, pass `useWallClockTime = false`: the filter then takes the time step only from `Observation.timestamp`. With the default `true` it also checks the system clock, which suits a live stream but makes a fast replay depend on how quickly the fixes are processed.

### Public API

- `AdaptiveKalmanFilterSdk` - ready-to-use facade for Android and custom sensor sources.
- `KalmanSdkConfig` - GPS, filter and smoother configuration.
- `SmootherConfig` - editable neural smoother parameters, including `windowLength` (`L`) and turn suppression.
- `Observation` - one GPS/IMU measurement consumed by the filter.
- `SensorDataSource` - platform-independent stream of observations.
- `AndroidSensorDataSource` - Android GPS/IMU implementation.
- `KalmanFilter` - lower-level stateful Kalman filter. After 3 fixes rejected in a row by the outlier gate it accepts the next one, so it cannot lose the track for good.
- `KalmanFilterCoordinator` - session runner that exposes `Flow<FilterResult>`.
- `CoefficientStrategy` - measurement-noise strategy interface; `KalmanFilter` passes it the innovation of every accepted fix through `onInnovation`.
- `ClassicalCoefficientStrategy` - default adaptive measurement-noise strategy.
- `MetricsEvaluator` - trajectory quality metrics; points are paired by timestamp (`TimedPosition`), so a delayed smoother is scored correctly.
- `NeuralTrajectorySmoother`, `SmoothingTrainingCollector`, `SmootherRepository` - optional neural smoothing pipeline.

### Benchmarks

The `benchmarks` module compares the engine with other filters on synthetic routes: a pedestrian and a bike, with white GPS noise, realistic noise (slow drift, multipath jumps, dropouts) and a phone that underestimates its accuracy in the city. Every method is tuned on 20 routes and tested on 40 others. It needs only a JDK, no Android SDK.

```bash
./gradlew :benchmarks:run
```

Without arguments it runs every table; pass names to run a part, and `help` for the list:

```bash
./gradlew :benchmarks:run --args="stage1 summary"
```

Results go to `benchmarks/build/results/`: `<experiment>.txt` with the tables and `<experiment>.csv` with every metric in long form (`table, scenario, method, lag_s, metric, value, params`). `--args="export"` also writes point-by-point tracks for plots and the whole synthetic dataset. All numbers come from simulated data and still need checking on real phone logs.

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

benchmarks
  Исследовательские бенчмарки, не публикуются: симулятор GPS и сенсоров
  телефона, конкуренты EKF/UKF и RTS, таблицы результатов и выгрузка CSV.
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

`SmootherConfig` управляет fixed-lag нейросетевым сглаживателем. Значения по умолчанию: `windowLength = 11`, `turnThresholdDeg = 30.0`, `alphaSuppressionTurnDeg = 60.0`, `minTurnSuppression = 0.05`, `turnDetection = TurnDetection.VELOCITY`, `minTurnSpeedMs = 0.5`.

Повороты определяются по направлению скорости из фильтра Калмана; точки медленнее `minTurnSpeedMs` пропускаются. Прежний режим `TurnDetection.POINT_DIFF` (направление между соседними отфильтрованными точками) находил «повороты» почти в каждом окне при ходьбе: на пешеходной скорости шаг за секунду сравним с дрожанием фильтра, и из-за этого почти все обучающие метки становились нулевыми.

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

Используйте один и тот же `SmootherConfig` при обучении и применении модели. Новые сохранённые модели (`v4`) хранят этот конфиг. Модели `v3` сохраняют свои значения, `v2` загружаются со значениями по умолчанию; обе загружаются с `TurnDetection.POINT_DIFF`, чтобы их признаки не изменились. Чтобы получить исправленное определение поворотов, переобучите их.

### Свой Источник Данных

Используйте свой `SensorDataSource`, если хост-приложение уже само получает GPS, проигрывает записанный маршрут, симулирует данные или работает с серверным источником.

```kotlin
import com.katka.data.ReplaySensorDataSource
import com.katka.sdk.AdaptiveKalmanFilterSdk
import com.katka.sdk.KalmanSdkConfig

val kalmanSdk = AdaptiveKalmanFilterSdk.create(
    sensorSource = ReplaySensorDataSource(observations),
    scope = scope,
    config = KalmanSdkConfig(useWallClockTime = false)
)
```

Для записанного маршрута передайте `useWallClockTime = false`: тогда шаг времени фильтр берёт только из `Observation.timestamp`. При значении по умолчанию `true` он сверяется ещё и с системными часами. Для живого потока это подходит, но при быстром воспроизведении результат зависит от того, как быстро обрабатываются фиксы.

### Публичный API

- `AdaptiveKalmanFilterSdk` - готовый facade для Android и пользовательских источников данных.
- `KalmanSdkConfig` - конфигурация GPS, фильтра и сглаживателя.
- `SmootherConfig` - редактируемые параметры нейросетевого сглаживателя, включая `windowLength` (`L`) и подавление на поворотах.
- `Observation` - одно GPS/IMU-измерение для фильтра.
- `SensorDataSource` - платформенно-независимый поток наблюдений.
- `AndroidSensorDataSource` - Android-реализация GPS/IMU-источника.
- `KalmanFilter` - низкоуровневый stateful фильтр Калмана. После 3 фиксов подряд, отброшенных гейтом выбросов, он принимает следующий и поэтому не может потерять трек навсегда.
- `KalmanFilterCoordinator` - runner сессии, который отдаёт `Flow<FilterResult>`.
- `CoefficientStrategy` - интерфейс стратегии шума измерений; `KalmanFilter` передаёт ей инновацию каждого принятого фикса через `onInnovation`.
- `ClassicalCoefficientStrategy` - стандартная адаптивная стратегия шума измерений.
- `MetricsEvaluator` - метрики качества траектории; точки сопоставляются по timestamp (`TimedPosition`), поэтому сглаживатель с задержкой оценивается правильно.
- `NeuralTrajectorySmoother`, `SmoothingTrainingCollector`, `SmootherRepository` - нейросетевая pipeline сглаживания.

### Бенчмарки

Модуль `benchmarks` сравнивает движок с другими фильтрами на синтетических маршрутах: пешеход и велосипед, белый шум GPS, реалистичный шум (медленный дрейф, скачки из-за многолучёвости, пропуски) и телефон, который в городе занижает свою погрешность. Параметры каждого метода подбираются на 20 маршрутах, проверка — на 40 других. Нужен только JDK, Android SDK не требуется.

```bash
./gradlew :benchmarks:run
```

Без аргументов считаются все таблицы. Чтобы запустить часть, перечислите имена; список выводит `help`:

```bash
./gradlew :benchmarks:run --args="stage1 summary"
```

Результаты пишутся в `benchmarks/build/results/`: `<эксперимент>.txt` с таблицами и `<эксперимент>.csv` со всеми метриками в длинном формате (`table, scenario, method, lag_s, metric, value, params`). `--args="export"` дополнительно выгружает треки по точкам для графиков и весь синтетический датасет. Все цифры получены на симуляции, их ещё нужно проверить на реальных записях с телефона.

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
