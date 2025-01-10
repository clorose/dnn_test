// path: predictor/src/main/kotlin/com/cnc/predictor/model/PredictionRequest.kt
package com.cnc.predictor.model

data class PredictionRequest(
    val features: List<Double>
)