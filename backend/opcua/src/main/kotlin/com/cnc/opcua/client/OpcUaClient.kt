package com.cnc.opcua.client

import org.eclipse.milo.opcua.sdk.client.OpcUaClient
import org.eclipse.milo.opcua.stack.core.security.SecurityPolicy
import org.springframework.stereotype.Component

@Component
class OpcUaClient {
    private lateinit var client: OpcUaClient

    fun connect() {
        try {
            client = OpcUaClient.create(
                "opc.tcp://localhost:12686",
                { endpoints ->
                    endpoints.stream()
                        .filter { e -> e.securityPolicyUri == SecurityPolicy.None.uri }
                        .findFirst()
                },
                { configBuilder -> configBuilder.build() }
            )
            
            client.connect().get()
            println("Connected to OPC UA server")
        } catch (ex: Exception) {
            println("Failed to connect: ${ex.message}")
            throw RuntimeException("Failed to connect to OPC UA server", ex)
        }
    }

    fun disconnect() {
        if (::client.isInitialized) {
            client.disconnect().get()
        }
    }
}