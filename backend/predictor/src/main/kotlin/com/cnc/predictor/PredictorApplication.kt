// path: predictor/src/main/kotlin/com/cnc/predictor/PredictorApplication.kt
package com.cnc.predictor

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class PredictorApplication

fun main(args: Array<String>) {
    runApplication<PredictorApplication>(*args)
}
