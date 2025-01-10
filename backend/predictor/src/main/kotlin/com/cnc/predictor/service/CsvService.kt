package com.cnc.predictor.service

import com.cnc.predictor.model.ProcessData
import org.springframework.stereotype.Service
import java.io.File
import com.opencsv.CSVReader
import java.io.FileReader
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.slf4j.LoggerFactory
import java.util.Locale

@Service
class CsvService {
    private val logger = LoggerFactory.getLogger(javaClass)
    private var dataList: MutableList<ProcessData> = mutableListOf()
    private var currentIndex = 0

    private val projectRoot = System.getProperty("user.dir").split("/backend")[0]
    private val specificFile = "$projectRoot/data/CNC Virtual Data set _v2/experiment_02.csv"

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
        loadCsvData()
    }

    private fun loadCsvData() {
        try {
            val file = File(specificFile)
            if (!file.exists()) {
                logger.error("CSV file not found: $specificFile")
                return
            }

            // val csvFiles = directory.listFiles { file -> file.extension == "csv" }
            // if (csvFiles.isNullOrEmpty()) {
            //     logger.error("No CSV files found in data directory: ${directory.absolutePath}")
            //     return
            // }

            // csvFiles.forEach { file ->
                try {
                    CSVReader(FileReader(file)).use { reader ->
                        reader.skip(1)
                        
                        reader.readAll().forEach { line ->
                            try {
                                val data = ProcessData(
                                    X_ActualPosition = line[0].toDouble(),
                                    X_ActualVelocity = line[1].toDouble(),
                                    X_ActualAcceleration = line[2].toDouble(),
                                    X_SetPosition = line[3].toDouble(),
                                    X_SetVelocity = line[4].toDouble(),
                                    X_SetAcceleration = line[5].toDouble(),
                                    X_CurrentFeedback = line[6].toDouble(),
                                    X_DCBusVoltage = line[7].toDouble(),
                                    X_OutputCurrent = line[8].toInt(),
                                    X_OutputVoltage = line[9].toDouble(),
                                    X_OutputPower = line[10].toDouble(),

                                    Y_ActualPosition = line[11].toDouble(),
                                    Y_ActualVelocity = line[12].toDouble(),
                                    Y_ActualAcceleration = line[13].toDouble(),
                                    Y_SetPosition = line[14].toDouble(),
                                    Y_SetVelocity = line[15].toDouble(),
                                    Y_SetAcceleration = line[16].toDouble(),
                                    Y_CurrentFeedback = line[17].toDouble(),
                                    Y_DCBusVoltage = line[18].toDouble(),
                                    Y_OutputCurrent = line[19].toInt(),
                                    Y_OutputVoltage = line[20].toDouble(),
                                    Y_OutputPower = line[21].toDouble(),

                                    Z_ActualPosition = line[22].toDouble(),
                                    Z_ActualVelocity = line[23].toDouble(),
                                    Z_ActualAcceleration = line[24].toDouble(),
                                    Z_SetPosition = line[25].toDouble(),
                                    Z_SetVelocity = line[26].toDouble(),
                                    Z_SetAcceleration = line[27].toDouble(),
                                    Z_CurrentFeedback = line[28].toDouble(),
                                    Z_DCBusVoltage = line[29].toDouble(),
                                    Z_OutputCurrent = line[30].toInt(),
                                    Z_OutputVoltage = line[31].toDouble(),

                                    S_ActualPosition = line[32].toDouble(),
                                    S_ActualVelocity = line[33].toDouble(),
                                    S_ActualAcceleration = line[34].toDouble(),
                                    S_SetPosition = line[35].toDouble(),
                                    S_SetVelocity = line[36].toDouble(),
                                    S_SetAcceleration = line[37].toDouble(),
                                    S_CurrentFeedback = line[38].toDouble(),
                                    S_DCBusVoltage = line[39].toDouble(),
                                    S_OutputCurrent = line[40].toInt(),
                                    S_OutputVoltage = line[41].toDouble(),
                                    S_OutputPower = line[42].toDouble(),
                                    S_SystemInertia = line[43].toDouble(),

                                    M_CURRENT_PROGRAM_NUMBER = line[44],
                                    M_sequence_number = line[45],
                                    M_CURRENT_FEEDRATE = line[46],
                                    Machining_Process = line[47]
                                )
                                dataList.add(data)
                            } catch (e: Exception) {
                                logger.error("Failed to parse CSV line in file: ${file.name}, error: ${e.message}")
                            }
                        }
                    }
                } catch (e: Exception) {
                    logger.error("Failed to process file: ${file.name}, error: ${e.message}")
                }
            // }
            logger.info("Successfully loaded ${dataList.size} records from CSV files")
        } catch (e: Exception) {
            logger.error("Failed to load CSV data: ${e.message}")
        }
    }

    fun getNextData(): ProcessData? {
        if (dataList.isEmpty()) {
            logger.warn("No data available")
            return null
        }
        return try {
            val data = dataList[currentIndex]
            currentIndex = (currentIndex + 1) % dataList.size
            data
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
        currentIndex = 0
        loadCsvData()
    }
}