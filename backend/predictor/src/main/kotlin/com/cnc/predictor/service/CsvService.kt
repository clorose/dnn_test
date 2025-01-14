// path: predictor/src/main/kotlin/com/cnc/predictor/service/CsvService.kt
package com.cnc.predictor.service

import com.cnc.predictor.model.ProcessData
import org.springframework.stereotype.Service
import java.io.File
import com.opencsv.CSVReader
import java.io.FileReader
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.slf4j.LoggerFactory

@Service
class CsvService {
    private val logger = LoggerFactory.getLogger(javaClass)
    private var dataList: MutableList<ProcessData> = mutableListOf()
    private var currentIndex = 0

    private val projectRoot = System.getProperty("user.dir").split("/backend")[0]
    private val dataDirectory = "$projectRoot/data/CNC Virtual Data set _v2"
    private val targetFiles = listOf(
        "experiment_02.csv",
        "experiment_07.csv",
        "experiment_16.csv",
        "experiment_17.csv",
        "experiment_19.csv",
        "experiment_20.csv",
        "experiment_22.csv",
        "experiment_25.csv"
    )

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
            val directory = File(dataDirectory)
            if (!directory.exists()) {
                logger.error("Data directory not found: $dataDirectory")
                return
            }

            targetFiles.forEach { fileName ->
                try {
                    val file = File(directory, fileName)
                    if (!file.exists()) {
                        logger.error("File not found: ${file.absolutePath}")
                        return@forEach
                    }

                    CSVReader(FileReader(file)).use { reader ->
                        var lineCount = 0
                        
                        // 헤더 읽고 스킵
                        reader.readNext()
                        lineCount++

                        // 데이터 행들 읽기
                        var line: Array<String>?
                        while (reader.readNext().also { line = it } != null) {
                            lineCount++
                            line?.let { csvLine ->
                                try {
                                    val data = ProcessData(
                                        X_ActualPosition = csvLine[0].toDouble(),
                                        X_ActualVelocity = csvLine[1].toDouble(),
                                        X_ActualAcceleration = csvLine[2].toDouble(),
                                        X_SetPosition = csvLine[3].toDouble(),
                                        X_SetVelocity = csvLine[4].toDouble(),
                                        X_SetAcceleration = csvLine[5].toDouble(),
                                        X_CurrentFeedback = csvLine[6].toDouble(),
                                        X_DCBusVoltage = csvLine[7].toDouble(),
                                        X_OutputCurrent = csvLine[8].toInt(),
                                        X_OutputVoltage = csvLine[9].toDouble(),
                                        X_OutputPower = csvLine[10].toDouble(),

                                        Y_ActualPosition = csvLine[11].toDouble(),
                                        Y_ActualVelocity = csvLine[12].toDouble(),
                                        Y_ActualAcceleration = csvLine[13].toDouble(),
                                        Y_SetPosition = csvLine[14].toDouble(),
                                        Y_SetVelocity = csvLine[15].toDouble(),
                                        Y_SetAcceleration = csvLine[16].toDouble(),
                                        Y_CurrentFeedback = csvLine[17].toDouble(),
                                        Y_DCBusVoltage = csvLine[18].toDouble(),
                                        Y_OutputCurrent = csvLine[19].toInt(),
                                        Y_OutputVoltage = csvLine[20].toDouble(),
                                        Y_OutputPower = csvLine[21].toDouble(),

                                        Z_ActualPosition = csvLine[22].toDouble(),
                                        Z_ActualVelocity = csvLine[23].toDouble(),
                                        Z_ActualAcceleration = csvLine[24].toDouble(),
                                        Z_SetPosition = csvLine[25].toDouble(),
                                        Z_SetVelocity = csvLine[26].toDouble(),
                                        Z_SetAcceleration = csvLine[27].toDouble(),
                                        Z_CurrentFeedback = csvLine[28].toDouble(),
                                        Z_DCBusVoltage = csvLine[29].toDouble(),
                                        Z_OutputCurrent = csvLine[30].toInt(),
                                        Z_OutputVoltage = csvLine[31].toDouble(),

                                        S_ActualPosition = csvLine[32].toDouble(),
                                        S_ActualVelocity = csvLine[33].toDouble(),
                                        S_ActualAcceleration = csvLine[34].toDouble(),
                                        S_SetPosition = csvLine[35].toDouble(),
                                        S_SetVelocity = csvLine[36].toDouble(),
                                        S_SetAcceleration = csvLine[37].toDouble(),
                                        S_CurrentFeedback = csvLine[38].toDouble(),
                                        S_DCBusVoltage = csvLine[39].toDouble(),
                                        S_OutputCurrent = csvLine[40].toInt(),
                                        S_OutputVoltage = csvLine[41].toDouble(),
                                        S_OutputPower = csvLine[42].toDouble(),
                                        S_SystemInertia = csvLine[43].toDouble(),

                                        M_CURRENT_PROGRAM_NUMBER = csvLine[44],
                                        M_sequence_number = csvLine[45],
                                        M_CURRENT_FEEDRATE = csvLine[46],
                                        Machining_Process = csvLine[47]
                                    )
                                    dataList.add(data)
                                } catch (e: Exception) {
                                    logger.error("Error parsing line $lineCount in file $fileName: ${e.message}")
                                }
                            }
                        }
                        logger.info("File: $fileName - Total lines: $lineCount (including header)")
                    }

                } catch (e: Exception) {
                    logger.error("Failed to process file: $fileName, error: ${e.message}")
                    e.printStackTrace()
                }
            }

            logger.info("Successfully loaded ${dataList.size} records from all CSV files")
        } catch (e: Exception) {
            logger.error("Failed to load CSV data: ${e.message}")
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
        currentIndex = 0
        loadCsvData()
    }
}