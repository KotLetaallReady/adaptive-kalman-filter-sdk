package com.katka.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class MatrixOpsTest {

    @Test
    fun inverse_ofRegularMatrix_givesIdentity() {
        val a = arrayOf(
            doubleArrayOf(4.0, 1.0, 0.5),
            doubleArrayOf(2.0, 3.0, 1.0),
            doubleArrayOf(0.0, 1.0, 2.0)
        )

        val product = MatrixOps.mul(a, MatrixOps.inverse(a))

        for (i in 0 until 3) for (j in 0 until 3) {
            assertEquals(if (i == j) 1.0 else 0.0, product[i][j], 1e-12)
        }
    }

    @Test
    fun inverse_ofSingularMatrix_staysFinite() {
        val singular = listOf(
            arrayOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 1.0)),
            MatrixOps.zeros(2, 2),
            // Covariance of a state whose velocity is known exactly.
            MatrixOps.diagonal(doubleArrayOf(4.0, 4.0, 0.0, 0.0))
        )

        for (m in singular) {
            val inv = MatrixOps.inverse(m)
            assertTrue(
                "inverse of ${m.contentDeepToString()} is ${inv.contentDeepToString()}",
                inv.all { row -> row.all { it.isFinite() } }
            )
        }
    }
}
