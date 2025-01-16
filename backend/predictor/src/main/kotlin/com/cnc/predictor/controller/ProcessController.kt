// path: predictor/src/main/kotlin/com/cnc/predictor/controller/ProcessController.kt
package com.cnc.predictor.controller

import com.cnc.predictor.service.CsvService
import com.cnc.predictor.model.PredictionResult
import com.cnc.predictor.model.PredictionRequest
import org.springframework.web.bind.annotation.*
import org.springframework.web.reactive.function.client.WebClient
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.slf4j.LoggerFactory
import java.util.Locale

@RestController
@RequestMapping("/api")
class ProcessController(private val csvService: CsvService) {
    private val logger = LoggerFactory.getLogger(javaClass)
    private val client = WebClient.builder().baseUrl("http://localhost:8000").defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE).build()
    
    @GetMapping("/process-data", produces = [MediaType.APPLICATION_JSON_VALUE])
    fun getProcessData(): ResponseEntity<Map<String, Any>> {
        val processData = csvService.getNextData() ?: return ResponseEntity.ok(mapOf(
                "status" to "error",
                "message" to "데이터를 불러올 수 없습니다."
            ))
    
        val features = csvService.toDoubleList(processData)
    
        val prediction = requestPrediction(features)  // scaledFeatures -> features
        
        return ResponseEntity.ok(mapOf(
            "status" to "success",
            "data" to processData,
            "prediction" to prediction
        ))
    }
    
    private fun requestPrediction(features: List<Double>): String {
        return try {
            logger.info("Sending prediction request with ${features.size} features")
            logger.info("Features: $features")
            
            val request = PredictionRequest(features = features)
            val response = client.post()
                .uri("/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PredictionResult::class.java)
                .block()
                ?: return "예측 실패: 응답 없음"

            response.prediction.firstOrNull()?.firstOrNull()?.toString() ?: "예측 실패: 결과 없음"
        } catch (e: Exception) {
            logger.error("Prediction failed: ${e.message}")
            "예측 실패: ${e.message}"
        }
    }
    
    @PostMapping("/reload-data")
    fun reloadData(): ResponseEntity<Map<String, String>> {
        return try {
            csvService.reloadData()
            ResponseEntity.ok(mapOf(
                "status" to "success",
                "message" to "데이터를 성공적으로 새로 불러왔습니다."
            ))
        } catch (e: Exception) {
            ResponseEntity.ok(mapOf(
                "status" to "error",
                "message" to "데이터 새로고침 실패: ${e.message}"
            ))
        }
    }
}