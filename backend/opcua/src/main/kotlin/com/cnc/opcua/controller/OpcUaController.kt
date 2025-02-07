package com.cnc.opcua.controller

import org.slf4j.LoggerFactory
import org.springframework.web.bind.annotation.*
import org.eclipse.milo.opcua.stack.core.types.builtin.DataValue
import com.cnc.opcua.client.OpcUaClient
import com.cnc.opcua.dto.*
import org.springframework.beans.factory.annotation.Autowired

@RestController
@RequestMapping("/api")
class OpcUaController {
    private val logger = LoggerFactory.getLogger(javaClass)
    
    @Autowired
    private lateinit var client: OpcUaClient

    @GetMapping("/status")
    fun getStatus(): Map<String, Any> {
        return mapOf(
            "connected" to client.getConnectionStatus(),
            "timestamp" to System.currentTimeMillis()
        )
    }

    @GetMapping("/data")
    fun getAllData(): ProcessDataDto {
        return try {
            val latestData = client.getLatestData()
            logger.info("Latest data size: ${latestData.size}")
            
            val missingKeys = findMissingKeys(latestData)
            if (missingKeys.isNotEmpty()) {
                logger.error("Missing keys in data: $missingKeys")
                throw IllegalStateException("Missing required data: ${missingKeys.joinToString()}")
            }
            
            createProcessDataDto(latestData)
        } catch (e: Exception) {
            logger.error("Error getting data: ${e.message}", e)
            throw e
        }
    }

    private fun findMissingKeys(data: Map<String, DataValue>): List<String> {
        val requiredKeys = setOf(
            "X_ActualPosition", "X_ActualVelocity", "X_ActualAcceleration",
            "X_SetPosition", "X_SetVelocity", "X_SetAcceleration",
            "X_CurrentFeedback", "X_DCBusVoltage", "X_OutputCurrent",
            "X_OutputVoltage", "X_OutputPower",
            
            "Y_ActualPosition", "Y_ActualVelocity", "Y_ActualAcceleration",
            "Y_SetPosition", "Y_SetVelocity", "Y_SetAcceleration",
            "Y_CurrentFeedback", "Y_DCBusVoltage", "Y_OutputCurrent",
            "Y_OutputVoltage", "Y_OutputPower",
            
            "Z_ActualPosition", "Z_ActualVelocity", "Z_ActualAcceleration",
            "Z_SetPosition", "Z_SetVelocity", "Z_SetAcceleration",
            "Z_CurrentFeedback", "Z_DCBusVoltage", "Z_OutputCurrent",
            "Z_OutputVoltage",
            
            "S_ActualPosition", "S_ActualVelocity", "S_ActualAcceleration",
            "S_SetPosition", "S_SetVelocity", "S_SetAcceleration",
            "S_CurrentFeedback", "S_DCBusVoltage", "S_OutputCurrent",
            "S_OutputVoltage", "S_OutputPower", "S_SystemInertia",
            
            "M_CURRENT_PROGRAM_NUMBER", "M_sequence_number",
            "M_CURRENT_FEEDRATE", "Machining_Process"
        )
        
        return requiredKeys.filter { !data.containsKey(it) }
    }

    @GetMapping("/debug")
    fun getDebugInfo(): Map<String, Any> {
        val latestData = client.getLatestData()
        return mapOf(
            "availableKeys" to latestData.keys.toList(),
            "dataValues" to latestData.mapValues { it.value.value.value },
            "connectionStatus" to client.getConnectionStatus(),
            "dataItemsCount" to latestData.size,
            "missingKeys" to findMissingKeys(latestData)
        )
    }

    @GetMapping("/data/all")
    fun getAllAxisData() = getAllData()

    private fun createProcessDataDto(data: Map<String, DataValue>): ProcessDataDto {
        fun getValue(key: String): Any {
            return data[key]?.value?.value
                ?: throw IllegalStateException("Missing value for $key")
        }

        return ProcessDataDto(
            x_ActualPosition = (getValue("X_ActualPosition") as Number).toDouble(),
            x_ActualVelocity = (getValue("X_ActualVelocity") as Number).toDouble(),
            x_ActualAcceleration = (getValue("X_ActualAcceleration") as Number).toDouble(),
            x_SetPosition = (getValue("X_SetPosition") as Number).toDouble(),
            x_SetVelocity = (getValue("X_SetVelocity") as Number).toDouble(),
            x_SetAcceleration = (getValue("X_SetAcceleration") as Number).toDouble(),
            x_CurrentFeedback = (getValue("X_CurrentFeedback") as Number).toDouble(),
            x_DCBusVoltage = (getValue("X_DCBusVoltage") as Number).toDouble(),
            x_OutputCurrent = (getValue("X_OutputCurrent") as Number).toInt(),
            x_OutputVoltage = (getValue("X_OutputVoltage") as Number).toDouble(),
            x_OutputPower = (getValue("X_OutputPower") as Number).toDouble(),
            
            y_ActualPosition = (getValue("Y_ActualPosition") as Number).toDouble(),
            y_ActualVelocity = (getValue("Y_ActualVelocity") as Number).toDouble(),
            y_ActualAcceleration = (getValue("Y_ActualAcceleration") as Number).toDouble(),
            y_SetPosition = (getValue("Y_SetPosition") as Number).toDouble(),
            y_SetVelocity = (getValue("Y_SetVelocity") as Number).toDouble(),
            y_SetAcceleration = (getValue("Y_SetAcceleration") as Number).toDouble(),
            y_CurrentFeedback = (getValue("Y_CurrentFeedback") as Number).toDouble(),
            y_DCBusVoltage = (getValue("Y_DCBusVoltage") as Number).toDouble(),
            y_OutputCurrent = (getValue("Y_OutputCurrent") as Number).toInt(),
            y_OutputVoltage = (getValue("Y_OutputVoltage") as Number).toDouble(),
            y_OutputPower = (getValue("Y_OutputPower") as Number).toDouble(),
            
            z_ActualPosition = (getValue("Z_ActualPosition") as Number).toDouble(),
            z_ActualVelocity = (getValue("Z_ActualVelocity") as Number).toDouble(),
            z_ActualAcceleration = (getValue("Z_ActualAcceleration") as Number).toDouble(),
            z_SetPosition = (getValue("Z_SetPosition") as Number).toDouble(),
            z_SetVelocity = (getValue("Z_SetVelocity") as Number).toDouble(),
            z_SetAcceleration = (getValue("Z_SetAcceleration") as Number).toDouble(),
            z_CurrentFeedback = (getValue("Z_CurrentFeedback") as Number).toDouble(),
            z_DCBusVoltage = (getValue("Z_DCBusVoltage") as Number).toDouble(),
            z_OutputCurrent = (getValue("Z_OutputCurrent") as Number).toInt(),
            z_OutputVoltage = (getValue("Z_OutputVoltage") as Number).toDouble(),
            
            s_ActualPosition = (getValue("S_ActualPosition") as Number).toDouble(),
            s_ActualVelocity = (getValue("S_ActualVelocity") as Number).toDouble(),
            s_ActualAcceleration = (getValue("S_ActualAcceleration") as Number).toDouble(),
            s_SetPosition = (getValue("S_SetPosition") as Number).toDouble(),
            s_SetVelocity = (getValue("S_SetVelocity") as Number).toDouble(),
            s_SetAcceleration = (getValue("S_SetAcceleration") as Number).toDouble(),
            s_CurrentFeedback = (getValue("S_CurrentFeedback") as Number).toDouble(),
            s_DCBusVoltage = (getValue("S_DCBusVoltage") as Number).toDouble(),
            s_OutputCurrent = (getValue("S_OutputCurrent") as Number).toInt(),
            s_OutputVoltage = (getValue("S_OutputVoltage") as Number).toDouble(),
            s_OutputPower = (getValue("S_OutputPower") as Number).toDouble(),
            s_SystemInertia = (getValue("S_SystemInertia") as Number).toDouble(),
            
            m_CURRENT_PROGRAM_NUMBER = getValue("M_CURRENT_PROGRAM_NUMBER").toString(),
            m_sequence_number = getValue("M_sequence_number").toString(),
            m_CURRENT_FEEDRATE = getValue("M_CURRENT_FEEDRATE").toString(),
            machining_Process = getValue("Machining_Process").toString()
        )
    }
}