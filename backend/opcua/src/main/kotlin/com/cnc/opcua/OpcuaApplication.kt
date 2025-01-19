package com.cnc.opcua

import com.cnc.opcua.server.OpcUaServer
import com.cnc.opcua.client.OpcUaClient
import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class OpcuaApplication

fun main(args: Array<String>) {
    val context = runApplication<OpcuaApplication>(*args)
    
    val server = context.getBean(OpcUaServer::class.java)
    val client = context.getBean(OpcUaClient::class.java)

    try {
        server.startup()
        client.connect()
        
        println("OPC-UA Server and Client connected successfully!")
        
    } catch (e: Exception) {
        println("Error: ${e.message}")
        e.printStackTrace()
    }
}