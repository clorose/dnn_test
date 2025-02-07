package com.cnc.opcua.client

import org.eclipse.milo.opcua.sdk.client.OpcUaClient
import org.eclipse.milo.opcua.sdk.client.subscriptions.ManagedSubscription
import org.eclipse.milo.opcua.stack.core.types.builtin.*
import org.eclipse.milo.opcua.stack.core.security.SecurityPolicy
import org.eclipse.milo.opcua.sdk.client.subscriptions.ManagedDataItem
import org.springframework.stereotype.Component
import java.util.concurrent.ConcurrentHashMap
import org.slf4j.LoggerFactory
import jakarta.annotation.PostConstruct
import jakarta.annotation.PreDestroy

@Component
class OpcUaClient {
    private val logger = LoggerFactory.getLogger(javaClass)
    private lateinit var client: OpcUaClient
    private lateinit var subscription: ManagedSubscription
    private val dataValues = ConcurrentHashMap<String, DataValue>()
    private val NAMESPACE_INDEX = 1
    private val dataItems = mutableListOf<ManagedDataItem>()
    
    @PostConstruct
    fun initialize() {
        connect()
    }

    @PreDestroy
    fun cleanup() {
        disconnect()
    }

    fun connect() {
        try {
            client = OpcUaClient.create(
                "opc.tcp://localhost:12686/milo",
                { endpoints ->
                    endpoints.stream()
                        .filter { e -> e.securityPolicyUri == SecurityPolicy.None.uri }
                        .findFirst()
                },
                { configBuilder -> configBuilder.build() }
            )
            
            client.connect().get()
            setupSubscription()
            logger.info("Connected to OPC UA server")
        } catch (ex: Exception) {
            logger.error("Failed to connect: ${ex.message}")
            throw RuntimeException("Failed to connect to OPC UA server", ex)
        }
    }

    private fun setupSubscription() {
        try {
            subscription = ManagedSubscription.create(client)
            
            subscription.addDataChangeListener { items, values ->
                for (i in items.indices) {
                    val nodeId = items[i].nodeId
                    val identifier = nodeId.identifier.toString()
                    val value = values[i]
                    logger.debug("Received data - NodeId: ${nodeId}, Value: ${value.value.value}")
                    dataValues[identifier] = value
                }
            }

            subscribeToAxisData()
        } catch (e: Exception) {
            logger.error("Failed to setup subscription: ${e.message}")
        }
    }
    
    private fun subscribeToAxisData() {
        setupXAxisSubscriptions()
        setupYAxisSubscriptions()
        setupZAxisSubscriptions()
        setupSpindleSubscriptions()
        setupMachiningSubscriptions()
    }

    private fun createSubscription(name: String) {
        try {
            val nodeId = NodeId(NAMESPACE_INDEX, name)
            val dataItem = subscription.createDataItem(nodeId)
            
            if (dataItem.statusCode.isGood) {
                dataItems.add(dataItem)
                logger.debug("Created subscription for: $name")
            } else {
                logger.error("Failed to create subscription for: $name, status: ${dataItem.statusCode}")
            }
        } catch (e: Exception) {
            logger.error("Error creating subscription for $name: ${e.message}")
        }
    }

    private fun setupXAxisSubscriptions() {
        listOf(
            "X_ActualPosition", "X_ActualVelocity", "X_ActualAcceleration",
            "X_SetPosition", "X_SetVelocity", "X_SetAcceleration",
            "X_CurrentFeedback", "X_DCBusVoltage", "X_OutputVoltage", 
            "X_OutputPower", "X_OutputCurrent"
        ).forEach { createSubscription(it) }
    }

    private fun setupYAxisSubscriptions() {
        listOf(
            "Y_ActualPosition", "Y_ActualVelocity", "Y_ActualAcceleration",
            "Y_SetPosition", "Y_SetVelocity", "Y_SetAcceleration",
            "Y_CurrentFeedback", "Y_DCBusVoltage", "Y_OutputVoltage", 
            "Y_OutputPower", "Y_OutputCurrent"
        ).forEach { createSubscription(it) }
    }

    private fun setupZAxisSubscriptions() {
        listOf(
            "Z_ActualPosition", "Z_ActualVelocity", "Z_ActualAcceleration",
            "Z_SetPosition", "Z_SetVelocity", "Z_SetAcceleration",
            "Z_CurrentFeedback", "Z_DCBusVoltage", "Z_OutputVoltage",
            "Z_OutputCurrent"
        ).forEach { createSubscription(it) }
    }

    private fun setupSpindleSubscriptions() {
        listOf(
            "S_ActualPosition", "S_ActualVelocity", "S_ActualAcceleration",
            "S_SetPosition", "S_SetVelocity", "S_SetAcceleration",
            "S_CurrentFeedback", "S_DCBusVoltage", "S_OutputVoltage",
            "S_OutputPower", "S_SystemInertia", "S_OutputCurrent"
        ).forEach { createSubscription(it) }
    }

    private fun setupMachiningSubscriptions() {
        listOf(
            "M_CURRENT_PROGRAM_NUMBER",
            "M_sequence_number",
            "M_CURRENT_FEEDRATE",
            "Machining_Process"
        ).forEach { createSubscription(it) }
    }

    fun getLatestData(): Map<String, DataValue> = HashMap(dataValues)

    fun getConnectionStatus(): Boolean = this::client.isInitialized

    fun disconnect() {
        dataItems.forEach { it.delete() }
        if (this::client.isInitialized) {
            try {
                client.disconnect().get()
            } catch (e: Exception) {
                logger.error("Error during disconnect: ${e.message}")
            }
        }
    }
}