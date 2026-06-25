package com.katka.android

import android.util.Log
import com.katka.engine.Logger

/**
 * [Logger] implementation that routes engine diagnostics to Android Logcat.
 *
 * @param tag Logcat tag used for every emitted message.
 */
class AndroidLogger(private val tag: String = "KalmanLog") : Logger {
    override fun d(message: String) { Log.d(tag, message) }
    override fun i(message: String) { Log.i(tag, message) }
    override fun w(message: String) { Log.w(tag, message) }
    override fun e(message: String, throwable: Throwable?) { Log.e(tag, message, throwable) }
}
