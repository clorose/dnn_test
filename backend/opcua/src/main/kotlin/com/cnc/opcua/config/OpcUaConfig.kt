package com.cnc.opcua.config

import com.cnc.opcua.server.OpcUaServer
import com.cnc.opcua.client.OpcUaClient
import org.springframework.context.annotation.Configuration
import org.springframework.boot.context.event.ApplicationReadyEvent
import org.springframework.context.event.EventListener
import org.springframework.beans.factory.annotation.Autowired

@Configuration
class OpcUaConfig {
    
    @Autowired
    private lateinit var server: OpcUaServer
    
    @Autowired
    private lateinit var client: OpcUaClient
    
    @EventListener(ApplicationReadyEvent::class)
    fun startOpcUa() {
        try {
            println("Starting OPC UA Server...")
            server.startup()
            
            Thread.sleep(1000)  // Give server time to start
            
            println("Connecting OPC UA Client...")
            client.connect()
        } catch (e: Exception) {
            println("Failed to initialize OPC UA: ${e.message}")
            e.printStackTrace()
        }
    }
}