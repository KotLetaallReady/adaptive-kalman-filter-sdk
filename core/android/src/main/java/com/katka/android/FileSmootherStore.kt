package com.katka.android

import android.content.Context
import com.katka.engine.neural.SmootherStore
import java.io.File

/**
 * Android file-backed [SmootherStore].
 *
 * The serialized model is stored in the application's private files directory,
 * so no external storage permission is required.
 *
 * @param context Android context used to resolve the private files directory.
 * @param fileName Name of the model file inside `context.filesDir`.
 */
class FileSmootherStore(
    private val context: Context,
    private val fileName: String = "neural_smoother.model"
) : SmootherStore {

    private fun file() = File(context.filesDir, fileName)

    override fun save(text: String) = file().writeText(text)

    override fun load(): String? = file().takeIf { it.exists() }?.readText()

    override fun exists(): Boolean = file().exists()

    override fun delete(): Boolean = file().delete()
}
