package com.cnc.opcua.server

import org.eclipse.milo.opcua.sdk.server.OpcUaServer
import org.eclipse.milo.opcua.sdk.server.api.config.OpcUaServerConfigBuilder
import org.eclipse.milo.opcua.stack.core.types.builtin.LocalizedText
import org.springframework.stereotype.Component

@Component
class OpcUaServer {
    private lateinit var server: OpcUaServer

    fun startup() {
        val config = OpcUaServerConfigBuilder()
            .setApplicationName(LocalizedText.english("CNC OPC-UA Test Server"))
            .setApplicationUri("urn:cnc:opcua:server")
            .setProductUri("urn:cnc:opcua:server")
            .build()

        server = OpcUaServer(config)
        server.startup().get()
        println("OPC UA Server started")
    }

    fun shutdown() {
        if (::server.isInitialized) {
            server.shutdown().get()
        }
    }
}