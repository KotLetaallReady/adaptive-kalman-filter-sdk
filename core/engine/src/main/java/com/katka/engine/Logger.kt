package com.katka.engine

/**
 * Logging boundary used by the engine.
 *
 * The pure Kotlin modules never depend on `android.util.Log`; Android, server
 * or test code can provide an implementation when diagnostics are needed.
 */
interface Logger {
    /** Writes a debug message. */
    fun d(message: String) {}

    /** Writes an informational message. */
    fun i(message: String) {}

    /** Writes a warning message. */
    fun w(message: String) {}

    /** Writes an error message and an optional cause. */
    fun e(message: String, throwable: Throwable? = null) {}

    /** Silent default logger. */
    object NoOp : Logger
}
