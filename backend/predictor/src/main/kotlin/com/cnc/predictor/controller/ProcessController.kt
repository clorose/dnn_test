package com.cnc.predictor.controller

import com.cnc.predictor.service.CsvService
import com.cnc.predictor.model.PredictionResult
import com.cnc.predictor.model.PredictionRequest
import org.springframework.web.bind.annotation.*
import org.springframework.web.reactive.function.client.WebClient
import org.springframework.http.*
import org.slf4j.LoggerFactory
import org.springframework.web.bind.annotation.CrossOrigin

@RestController
@RequestMapping("/api")
@CrossOrigin(
    origins = ["*"],
    allowedHeaders = ["*"],
    methods = [
        RequestMethod.GET,
        RequestMethod.POST,
        RequestMethod.PUT,
        RequestMethod.DELETE,
        RequestMethod.OPTIONS
    ]
)
class ProcessController(private val csvService: CsvService) {
    private val logger = LoggerFactory.getLogger(javaClass)
    private val client = WebClient.builder()
        .baseUrl("http://localhost:8000")
        .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
        .build()
    
    @GetMapping("/process-data")
    fun getProcessData(): ResponseEntity<Map<String, Any>> {
        return try {
            val processData = csvService.getNextData() ?: return ResponseEntity
                .status(HttpStatus.NOT_FOUND)
                .contentType(MediaType.APPLICATION_JSON)
                .body(mapOf(
                    "status" to "error",
                    "message" to "데이터를 불러올 수 없습니다."
                ))
    
            val features = csvService.toDoubleList(processData)
            val prediction = requestPrediction(features)
            
            ResponseEntity
                .ok()
                .contentType(MediaType.APPLICATION_JSON)
                .body(mapOf(
                    "status" to "success",
                    "data" to processData,
                    "prediction" to prediction
                ))
        } catch (e: Exception) {
            logger.error("Error processing data: ${e.message}", e)
            ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .contentType(MediaType.APPLICATION_JSON)
                .body(mapOf(
                    "status" to "error",
                    "message" to "처리 중 오류가 발생했습니다: ${e.message}"
                ))
        }
    }
    
    private fun requestPrediction(features: List<Double>): String {
        return try {
            logger.info("Sending prediction request with ${features.size} features")
            logger.debug("Features: $features")
            
            val request = PredictionRequest(features = features)
            val response = client.post()
                .uri("/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request)
                .retrieve()
                .bodyToMono(PredictionResult::class.java)
                .block()
                ?: return "예측 실패: 응답 없음"

            response.prediction.firstOrNull()?.firstOrNull()?.toString() 
                ?: "예측 실패: 결과 없음"
        } catch (e: Exception) {
            logger.error("Prediction failed: ${e.message}", e)
            "예측 실패: ${e.message}"
        }
    }
    
    @PostMapping("/reload-data")
    fun reloadData(): ResponseEntity<Map<String, String>> {
        return try {
            csvService.reloadData()
            ResponseEntity
                .ok()
                .contentType(MediaType.APPLICATION_JSON)
                .body(mapOf(
                    "status" to "success",
                    "message" to "데이터를 성공적으로 새로 불러왔습니다."
                ))
        } catch (e: Exception) {
            ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .contentType(MediaType.APPLICATION_JSON)
                .body(mapOf(
                    "status" to "error",
                    "message" to "데이터 새로고침 실패: ${e.message}"
                ))
        }
    }
}