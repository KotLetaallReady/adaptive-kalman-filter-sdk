package com.katka.engine

import kotlin.math.abs

/** Row-major matrix arithmetic for the filter. */
internal object MatrixOps {

    fun zeros(rows: Int, cols: Int): Array<DoubleArray> =
        Array(rows) { DoubleArray(cols) }

    fun identity(n: Int): Array<DoubleArray> =
        Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    fun diagonal(values: DoubleArray): Array<DoubleArray> {
        val n = values.size
        return Array(n) { i -> DoubleArray(n) { j -> if (i == j) values[i] else 0.0 } }
    }

    fun copy(A: Array<DoubleArray>): Array<DoubleArray> =
        Array(A.size) { i -> A[i].copyOf() }

    fun add(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        require(A.size == B.size && A[0].size == B[0].size) { "add: shape mismatch" }
        return Array(A.size) { i -> DoubleArray(A[0].size) { j -> A[i][j] + B[i][j] } }
    }

    fun sub(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        require(A.size == B.size && A[0].size == B[0].size) { "sub: shape mismatch" }
        return Array(A.size) { i -> DoubleArray(A[0].size) { j -> A[i][j] - B[i][j] } }
    }

    fun scale(A: Array<DoubleArray>, s: Double): Array<DoubleArray> =
        Array(A.size) { i -> DoubleArray(A[0].size) { j -> A[i][j] * s } }

    fun mul(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val n = A.size; val p = B[0].size; val k = B.size
        require(A[0].size == k) { "mul: inner dimensions don't match (${A[0].size} vs $k)" }
        return Array(n) { i ->
            DoubleArray(p) { j -> (0 until k).sumOf { l -> A[i][l] * B[l][j] } }
        }
    }

    fun mulVec(A: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        require(A[0].size == v.size) { "mulVec: shape mismatch" }
        return DoubleArray(A.size) { i -> (v.indices).sumOf { j -> A[i][j] * v[j] } }
    }

    fun outerProduct(a: DoubleArray, b: DoubleArray): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(b.size) { j -> a[i] * b[j] } }

    fun transpose(A: Array<DoubleArray>): Array<DoubleArray> {
        val n = A.size; val m = A[0].size
        return Array(m) { i -> DoubleArray(n) { j -> A[j][i] } }
    }

    fun addDiagEps(A: Array<DoubleArray>, eps: Double = 1e-9): Array<DoubleArray> {
        val R = copy(A)
        for (i in R.indices) R[i][i] += eps
        return R
    }

    /**
     * Gauss–Jordan inverse with partial pivoting. A near-zero pivot is nudged by `1e-6`, so a
     * singular matrix gives a large but finite result instead of failing.
     */
    fun inverse(A: Array<DoubleArray>): Array<DoubleArray> {
        val n = A.size
        require(n > 0 && A.all { it.size == n }) { "inverse: must be square" }

        val aug = Array(n) { i ->
            DoubleArray(2 * n) { j -> if (j < n) A[i][j] else if (j - n == i) 1.0 else 0.0 }
        }

        for (col in 0 until n) {
            var maxRow = col
            for (row in col + 1 until n) {
                if (abs(aug[row][col]) > abs(aug[maxRow][col])) maxRow = row
            }
            val tmp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = tmp

            if (abs(aug[col][col]) < 1e-10) {
                aug[col][col] += 1e-6
            }
            val pivot = aug[col][col]

            for (j in 0 until 2 * n) aug[col][j] /= pivot

            for (row in 0 until n) {
                if (row == col) continue
                val factor = aug[row][col]
                for (j in 0 until 2 * n) aug[row][j] -= factor * aug[col][j]
            }
        }

        return Array(n) { i -> DoubleArray(n) { j -> aug[i][j + n] } }
    }

    fun symmetrise(A: Array<DoubleArray>): Array<DoubleArray> {
        val n = A.size
        return Array(n) { i -> DoubleArray(n) { j -> (A[i][j] + A[j][i]) / 2.0 } }
    }

    fun Array<DoubleArray>.toDebugString(decimals: Int = 4): String {
        val fmt = "%.${decimals}f"
        return joinToString(prefix = "[\n", postfix = "\n]", separator = "\n") { row ->
            "  " + row.joinToString(" ") { fmt.format(it) }
        }
    }

    fun DoubleArray.toDebugString(decimals: Int = 4): String {
        val fmt = "%.${decimals}f"
        return "[" + joinToString(", ") { fmt.format(it) } + "]"
    }
}
