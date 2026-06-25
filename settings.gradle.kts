pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "AdaptiveKalmanFilterSdk"
include(":adaptive-kalman-filter-sdk")
project(":adaptive-kalman-filter-sdk").projectDir = file("sdk")
include(":core:model")
include(":core:engine")
include(":core:data")
include(":core:android")
