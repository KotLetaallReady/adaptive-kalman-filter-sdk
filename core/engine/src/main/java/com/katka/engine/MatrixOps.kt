package com.katka.engine

import kotlin.math.abs

/** Internal row-major matrix arithmetic used by the Kalman filter. */
internal object MatrixOps {

    /** Zero matrix of the given shape. */
    fun zeros(rows: Int, cols: Int): Array<DoubleArray> =
        Array(rows) { DoubleArray(cols) }

    /** n×n identity matrix. */
    fun identity(n: Int): Array<DoubleArray> =
        Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    /** Diagonal matrix from the given values. */
    fun diagonal(values: DoubleArray): Array<DoubleArray> {
        val n = values.size
        return Array(n) { i -> DoubleArray(n) { j -> if (i == j) values[i] else 0.0 } }
    }

    /** Deep copy of a matrix. */
    fun copy(A: Array<DoubleArray>): Array<DoubleArray> =
        Array(A.size) { i -> A[i].copyOf() }

    /** Element-wise sum A + B. */
    fun add(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        require(A.size == B.size && A[0].size == B[0].size) { "add: shape mismatch" }
        return Array(A.size) { i -> DoubleArray(A[0].size) { j -> A[i][j] + B[i][j] } }
    }

    /** Element-wise difference A − B. */
    fun sub(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        require(A.size == B.size && A[0].size == B[0].size) { "sub: shape mismatch" }
        return Array(A.size) { i -> DoubleArray(A[0].size) { j -> A[i][j] - B[i][j] } }
    }

    /** Scalar multiple s·A. */
    fun scale(A: Array<DoubleArray>, s: Double): Array<DoubleArray> =
        Array(A.size) { i -> DoubleArray(A[0].size) { j -> A[i][j] * s } }

    /** Matrix product A·B. */
    fun mul(A: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        val n = A.size; val p = B[0].size; val k = B.size
        require(A[0].size == k) { "mul: inner dimensions don't match (${A[0].size} vs $k)" }
        return Array(n) { i ->
            DoubleArray(p) { j -> (0 until k).sumOf { l -> A[i][l] * B[l][j] } }
        }
    }

    /** Matrix-vector product A·v. */
    fun mulVec(A: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        require(A[0].size == v.size) { "mulVec: shape mismatch" }
        return DoubleArray(A.size) { i -> (v.indices).sumOf { j -> A[i][j] * v[j] } }
    }

    /** Outer product a·bᵀ. */
    fun outerProduct(a: DoubleArray, b: DoubleArray): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(b.size) { j -> a[i] * b[j] } }

    /** Transpose Aᵀ. */
    fun transpose(A: Array<DoubleArray>): Array<DoubleArray> {
        val n = A.size; val m = A[0].size
        return Array(m) { i -> DoubleArray(n) { j -> A[j][i] } }
    }

    /** Returns a copy of A with a small epsilon added to the diagonal for numerical safety before inversion. */
    fun addDiagEps(A: Array<DoubleArray>, eps: Double = 1e-9): Array<DoubleArray> {
        val R = copy(A)
        for (i in R.indices) R[i][i] += eps
        return R
    }

    /** Matrix inverse via Gauss-Jordan elimination with partial pivoting; throws if singular. */
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

            val pivot = aug[col][col]
            if (abs(pivot) < 1e-10) {
                aug[col][col] += 1e-6
            }

            for (j in 0 until 2 * n) aug[col][j] /= pivot

            for (row in 0 until n) {
                if (row == col) continue
                val factor = aug[row][col]
                for (j in 0 until 2 * n) aug[row][j] -= factor * aug[col][j]
            }
        }

        return Array(n) { i -> DoubleArray(n) { j -> aug[i][j + n] } }
    }

    /** Forces exact symmetry: A = (A + Aᵀ) / 2. */
    fun symmetrise(A: Array<DoubleArray>): Array<DoubleArray> {
        val n = A.size
        return Array(n) { i -> DoubleArray(n) { j -> (A[i][j] + A[j][i]) / 2.0 } }
    }

    /** Pretty-prints a matrix for debugging. */
    fun Array<DoubleArray>.toDebugString(decimals: Int = 4): String {
        val fmt = "%.${decimals}f"
        return joinToString(prefix = "[\n", postfix = "\n]", separator = "\n") { row ->
            "  " + row.joinToString(" ") { fmt.format(it) }
        }
    }

    /** Pretty-prints a vector for debugging. */
    fun DoubleArray.toDebugString(decimals: Int = 4): String {
        val fmt = "%.${decimals}f"
        return "[" + joinToString(", ") { fmt.format(it) } + "]"
    }
}
