package com.cnc.opcua.controller

import org.slf4j.LoggerFactory  // 이 import 추가
import org.springframework.web.bind.annotation.*
import org.eclipse.milo.opcua.stack.core.types.builtin.DataValue
import org.springframework.stereotype.Controller
import com.cnc.opcua.server.OpcUaServer
import com.cnc.opcua.client.OpcUaClient
import org.springframework.beans.factory.annotation.Autowired

@RestController
@RequestMapping("/api")
class OpcUaController {

    private val logger = LoggerFactory.getLogger(javaClass)  // 로거 추가
    
    @Autowired
    private lateinit var server: OpcUaServer
    
    @Autowired
    private lateinit var client: OpcUaClient

    @GetMapping("/status")
    fun getStatus(): Map<String, Any> {
        return mapOf(
            "serverStatus" to "Running",
            "clientStatus" to "Connected",
            "timestamp" to System.currentTimeMillis()
        )
    }

    @GetMapping("/data")
    fun getAllData(): Map<String, Any> {
        return try {
            val latestData = client.getLatestData()
            logger.info("Latest data size: ${latestData.size}")  // 로그 추가
            latestData.mapValues { (key, value) ->
                logger.info("Processing key: $key")  // 로그 추가
                value.value.value
            }
        } catch (e: Exception) {
            logger.error("Error getting data: ${e.message}", e)  // 에러 로깅
            mapOf("error" to e.message.toString())
        }
    }

    @GetMapping("/data/axis")
    fun getAxisData(): Map<String, Map<String, Any>> {
        val allData = client.getLatestData()
        return mapOf(
            "xAxis" to filterAxisData(allData, "X"),
            "yAxis" to filterAxisData(allData, "Y"),
            "zAxis" to filterAxisData(allData, "Z"),
            "spindle" to filterAxisData(allData, "S"),
            "machining" to filterMachiningData(allData)
        )
    }

    private fun filterAxisData(data: Map<String, DataValue>, prefix: String): Map<String, Any> {
        return data.filter { (key, _) -> key.startsWith("${prefix}_") }
            .mapValues { (_, value) -> 
                when (val v = value.value.value) {
                    is Number -> v
                    is String -> v
                    else -> v.toString()
                }
            }
    }

    private fun filterMachiningData(data: Map<String, DataValue>): Map<String, Any> {
        return data.filter { (key, _) -> key.startsWith("M_") }
            .mapValues { (_, value) ->
                when (val v = value.value.value) {
                    is Number -> v
                    is String -> v
                    else -> v.toString()
                }
            }
    }

    @GetMapping("/data/axis/x")
    fun getXAxisData() = filterAxisData(client.getLatestData(), "X")

    @GetMapping("/data/axis/y")
    fun getYAxisData() = filterAxisData(client.getLatestData(), "Y")

    @GetMapping("/data/axis/z")
    fun getZAxisData() = filterAxisData(client.getLatestData(), "Z")

    @GetMapping("/data/spindle")
    fun getSpindleData() = filterAxisData(client.getLatestData(), "S")

    @GetMapping("/data/machining")
    fun getMachiningData() = filterMachiningData(client.getLatestData())
}