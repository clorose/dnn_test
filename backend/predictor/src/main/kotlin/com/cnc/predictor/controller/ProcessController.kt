package com.cnc.predictor.controller

import com.cnc.opcua.client.OpcUaClient
import com.cnc.predictor.model.ProcessData
import com.cnc.predictor.model.PredictionRequest
import com.cnc.predictor.model.PredictionResult
import org.eclipse.milo.opcua.stack.core.types.builtin.DataValue
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
class ProcessController(private val opcUaClient: OpcUaClient) {
    private val logger = LoggerFactory.getLogger(javaClass)
    private val client = WebClient.builder()
        .baseUrl("http://localhost:8000")
        .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
        .build()
    
    private val machiningProcessMap = mapOf(
        "Prep" to 0.0,
        "Layer 1 Up" to 1.0,
        "Layer 1 Down" to 2.0,
        "Layer 2 Up" to 3.0,
        "Layer 2 Down" to 4.0,
        "Layer 3 Up" to 5.0,
        "Layer 3 Down" to 6.0,
        "Repositioning" to 7.0,
        "End" to 8.0,
        "Starting" to 9.0
    )
    
    @GetMapping("/process-data")
    fun getProcessData(): ResponseEntity<Map<String, Any>> {
        return try {
            val rawData = opcUaClient.getLatestData()
            if (rawData.isEmpty()) {
                return ResponseEntity
                    .status(HttpStatus.NOT_FOUND)
                    .body(mapOf(
                        "status" to "error",
                        "message" to "데이터를 불러올 수 없습니다."
                    ))
            }

            val processData = convertToProcessData(rawData)
            val features = processDataToFeatures(processData)
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
                .body(mapOf(
                    "status" to "error",
                    "message" to "처리 중 오류가 발생했습니다: ${e.message}"
                ))
        }
    }
    
    private fun convertToProcessData(data: Map<String, DataValue>): ProcessData {
        fun getValue(key: String): Any {
            return data[key]?.value?.value
                ?: throw IllegalStateException("Missing value for $key")
        }

        return ProcessData(
            X_ActualPosition = (getValue("X_ActualPosition") as Number).toDouble(),
            X_ActualVelocity = (getValue("X_ActualVelocity") as Number).toDouble(),
            X_ActualAcceleration = (getValue("X_ActualAcceleration") as Number).toDouble(),
            X_SetPosition = (getValue("X_SetPosition") as Number).toDouble(),
            X_SetVelocity = (getValue("X_SetVelocity") as Number).toDouble(),
            X_SetAcceleration = (getValue("X_SetAcceleration") as Number).toDouble(),
            X_CurrentFeedback = (getValue("X_CurrentFeedback") as Number).toDouble(),
            X_DCBusVoltage = (getValue("X_DCBusVoltage") as Number).toDouble(),
            X_OutputCurrent = (getValue("X_OutputCurrent") as Number).toInt(),
            X_OutputVoltage = (getValue("X_OutputVoltage") as Number).toDouble(),
            X_OutputPower = (getValue("X_OutputPower") as Number).toDouble(),

            Y_ActualPosition = (getValue("Y_ActualPosition") as Number).toDouble(),
            Y_ActualVelocity = (getValue("Y_ActualVelocity") as Number).toDouble(),
            Y_ActualAcceleration = (getValue("Y_ActualAcceleration") as Number).toDouble(),
            Y_SetPosition = (getValue("Y_SetPosition") as Number).toDouble(),
            Y_SetVelocity = (getValue("Y_SetVelocity") as Number).toDouble(),
            Y_SetAcceleration = (getValue("Y_SetAcceleration") as Number).toDouble(),
            Y_CurrentFeedback = (getValue("Y_CurrentFeedback") as Number).toDouble(),
            Y_DCBusVoltage = (getValue("Y_DCBusVoltage") as Number).toDouble(),
            Y_OutputCurrent = (getValue("Y_OutputCurrent") as Number).toInt(),
            Y_OutputVoltage = (getValue("Y_OutputVoltage") as Number).toDouble(),
            Y_OutputPower = (getValue("Y_OutputPower") as Number).toDouble(),

            Z_ActualPosition = (getValue("Z_ActualPosition") as Number).toDouble(),
            Z_ActualVelocity = (getValue("Z_ActualVelocity") as Number).toDouble(),
            Z_ActualAcceleration = (getValue("Z_ActualAcceleration") as Number).toDouble(),
            Z_SetPosition = (getValue("Z_SetPosition") as Number).toDouble(),
            Z_SetVelocity = (getValue("Z_SetVelocity") as Number).toDouble(),
            Z_SetAcceleration = (getValue("Z_SetAcceleration") as Number).toDouble(),
            Z_CurrentFeedback = (getValue("Z_CurrentFeedback") as Number).toDouble(),
            Z_DCBusVoltage = (getValue("Z_DCBusVoltage") as Number).toDouble(),
            Z_OutputCurrent = (getValue("Z_OutputCurrent") as Number).toInt(),
            Z_OutputVoltage = (getValue("Z_OutputVoltage") as Number).toDouble(),

            S_ActualPosition = (getValue("S_ActualPosition") as Number).toDouble(),
            S_ActualVelocity = (getValue("S_ActualVelocity") as Number).toDouble(),
            S_ActualAcceleration = (getValue("S_ActualAcceleration") as Number).toDouble(),
            S_SetPosition = (getValue("S_SetPosition") as Number).toDouble(),
            S_SetVelocity = (getValue("S_SetVelocity") as Number).toDouble(),
            S_SetAcceleration = (getValue("S_SetAcceleration") as Number).toDouble(),
            S_CurrentFeedback = (getValue("S_CurrentFeedback") as Number).toDouble(),
            S_DCBusVoltage = (getValue("S_DCBusVoltage") as Number).toDouble(),
            S_OutputCurrent = (getValue("S_OutputCurrent") as Number).toInt(),
            S_OutputVoltage = (getValue("S_OutputVoltage") as Number).toDouble(),
            S_OutputPower = (getValue("S_OutputPower") as Number).toDouble(),
            S_SystemInertia = (getValue("S_SystemInertia") as Number).toDouble(),

            M_CURRENT_PROGRAM_NUMBER = getValue("M_CURRENT_PROGRAM_NUMBER").toString(),
            M_sequence_number = getValue("M_sequence_number").toString(),
            M_CURRENT_FEEDRATE = getValue("M_CURRENT_FEEDRATE").toString(),
            Machining_Process = getValue("Machining_Process").toString()
        )
    }

    private fun processDataToFeatures(data: ProcessData): List<Double> = listOf(
        data.X_ActualPosition, data.X_ActualVelocity, data.X_ActualAcceleration,
        data.X_SetPosition, data.X_SetVelocity, data.X_SetAcceleration,
        data.X_CurrentFeedback, data.X_DCBusVoltage, data.X_OutputCurrent.toDouble(),
        data.X_OutputVoltage, data.X_OutputPower,

        data.Y_ActualPosition, data.Y_ActualVelocity, data.Y_ActualAcceleration,
        data.Y_SetPosition, data.Y_SetVelocity, data.Y_SetAcceleration,
        data.Y_CurrentFeedback, data.Y_DCBusVoltage, data.Y_OutputCurrent.toDouble(),
        data.Y_OutputVoltage, data.Y_OutputPower,

        data.Z_ActualPosition, data.Z_ActualVelocity, data.Z_ActualAcceleration,
        data.Z_SetPosition, data.Z_SetVelocity, data.Z_SetAcceleration,
        data.Z_CurrentFeedback, data.Z_DCBusVoltage, data.Z_OutputCurrent.toDouble(),
        data.Z_OutputVoltage,

        data.S_ActualPosition, data.S_ActualVelocity, data.S_ActualAcceleration,
        data.S_SetPosition, data.S_SetVelocity, data.S_SetAcceleration,
        data.S_CurrentFeedback, data.S_DCBusVoltage, data.S_OutputCurrent.toDouble(),
        data.S_OutputVoltage, data.S_OutputPower, data.S_SystemInertia,

        data.M_CURRENT_PROGRAM_NUMBER.toDouble(),
        data.M_sequence_number.toDouble(),
        data.M_CURRENT_FEEDRATE.toDouble(),
        machiningProcessMap.getOrDefault(data.Machining_Process, 0.0)
    )
    
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
}