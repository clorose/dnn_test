package com.cnc.predictor.service

import com.cnc.predictor.model.ProcessData
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.stereotype.Service
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.slf4j.LoggerFactory

@Service
class MachineProcessService {
    @Autowired
    private lateinit var jdbcTemplate: JdbcTemplate

    private val logger = LoggerFactory.getLogger(javaClass)
    private var dataList: MutableList<ProcessData> = mutableListOf()

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

    @EventListener(ApplicationReadyEvent::class)
    fun init() {
        loadDataFromDatabase()
    }

    private fun loadDataFromDatabase() {
        try {
            val sql = """
                SELECT * FROM machine_process
            """.trimIndent()
            
            dataList = jdbcTemplate.query(sql) { rs, _ ->
                ProcessData(
                    X_ActualPosition = rs.getDouble("X1_ActualPosition"),
                    X_ActualVelocity = rs.getDouble("X1_ActualVelocity"),
                    X_ActualAcceleration = rs.getDouble("X1_ActualAcceleration"),
                    X_SetPosition = rs.getDouble("X1_CommandPosition"),
                    X_SetVelocity = rs.getDouble("X1_CommandVelocity"),
                    X_SetAcceleration = rs.getDouble("X1_CommandAcceleration"),
                    X_CurrentFeedback = rs.getDouble("X1_CurrentFeedback"),
                    X_DCBusVoltage = rs.getDouble("X1_DCBusVoltage"),
                    X_OutputCurrent = rs.getInt("X1_OutputCurrent"),
                    X_OutputVoltage = rs.getDouble("X1_OutputVoltage"),
                    X_OutputPower = rs.getDouble("X1_OutputPower"),
                    
                    Y_ActualPosition = rs.getDouble("Y1_ActualPosition"),
                    Y_ActualVelocity = rs.getDouble("Y1_ActualVelocity"),
                    Y_ActualAcceleration = rs.getDouble("Y1_ActualAcceleration"),
                    Y_SetPosition = rs.getDouble("Y1_CommandPosition"),
                    Y_SetVelocity = rs.getDouble("Y1_CommandVelocity"),
                    Y_SetAcceleration = rs.getDouble("Y1_CommandAcceleration"),
                    Y_CurrentFeedback = rs.getDouble("Y1_CurrentFeedback"),
                    Y_DCBusVoltage = rs.getDouble("Y1_DCBusVoltage"),
                    Y_OutputCurrent = rs.getInt("Y1_OutputCurrent"),
                    Y_OutputVoltage = rs.getDouble("Y1_OutputVoltage"),
                    Y_OutputPower = rs.getDouble("Y1_OutputPower"),
                    
                    Z_ActualPosition = rs.getDouble("Z1_ActualPosition"),
                    Z_ActualVelocity = rs.getDouble("Z1_ActualVelocity"),
                    Z_ActualAcceleration = rs.getDouble("Z1_ActualAcceleration"),
                    Z_SetPosition = rs.getDouble("Z1_CommandPosition"),
                    Z_SetVelocity = rs.getDouble("Z1_CommandVelocity"),
                    Z_SetAcceleration = rs.getDouble("Z1_CommandAcceleration"),
                    Z_CurrentFeedback = rs.getDouble("Z1_CurrentFeedback"),
                    Z_DCBusVoltage = rs.getDouble("Z1_DCBusVoltage"),
                    Z_OutputCurrent = rs.getInt("Z1_OutputCurrent"),
                    Z_OutputVoltage = rs.getDouble("Z1_OutputVoltage"),
                    
                    S_ActualPosition = rs.getDouble("S1_ActualPosition"),
                    S_ActualVelocity = rs.getDouble("S1_ActualVelocity"),
                    S_ActualAcceleration = rs.getDouble("S1_ActualAcceleration"),
                    S_SetPosition = rs.getDouble("S1_CommandPosition"),
                    S_SetVelocity = rs.getDouble("S1_CommandVelocity"),
                    S_SetAcceleration = rs.getDouble("S1_CommandAcceleration"),
                    S_CurrentFeedback = rs.getDouble("S1_CurrentFeedback"),
                    S_DCBusVoltage = rs.getDouble("S1_DCBusVoltage"),
                    S_OutputCurrent = rs.getInt("S1_OutputCurrent"),
                    S_OutputVoltage = rs.getDouble("S1_OutputVoltage"),
                    S_OutputPower = rs.getDouble("S1_OutputPower"),
                    S_SystemInertia = rs.getDouble("S1_SystemInertia"),
                    
                    M_CURRENT_PROGRAM_NUMBER = rs.getString("M1_CURRENT_PROGRAM_NUMBER"),
                    M_sequence_number = rs.getString("M1_sequence_number"),
                    M_CURRENT_FEEDRATE = rs.getString("M1_CURRENT_FEEDRATE"),
                    Machining_Process = rs.getString("Machining_Process")
                )
            }
            
            logger.info("Successfully loaded ${dataList.size} records from database")
        } catch (e: Exception) {
            logger.error("Failed to load data from database: ${e.message}")
            e.printStackTrace()
        }
    }

    fun getNextData(): ProcessData? {
        if (dataList.isEmpty()) {
            logger.warn("No data available")
            return null
        }
        return try {
            val randomIndex = (0 until dataList.size).random()
            dataList[randomIndex]
        } catch (e: Exception) {
            logger.error("Error getting next data: ${e.message}")
            null
        }
    }

    fun toDoubleList(data: ProcessData): List<Double> = listOf(
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

    fun getNextFeatures(): List<Double>? {
        return getNextData()?.let { data ->
            toDoubleList(data)
        }
    }

    fun reloadData() {
        dataList.clear()
        loadDataFromDatabase()
    }
}