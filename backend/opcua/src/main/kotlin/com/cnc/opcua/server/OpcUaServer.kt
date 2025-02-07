// path: test/backend/opcua/src/main/kotlin/com/cnc/opcua/server/OpcUaServer.kt
package com.cnc.opcua.server

import com.opencsv.CSVReader
import java.io.File
import java.io.FileReader
import kotlin.concurrent.fixedRateTimer
import org.slf4j.LoggerFactory

import org.eclipse.milo.opcua.sdk.core.AccessLevel
import org.eclipse.milo.opcua.sdk.core.Reference
import org.eclipse.milo.opcua.sdk.server.OpcUaServer
import org.eclipse.milo.opcua.sdk.server.api.DataItem
import org.eclipse.milo.opcua.sdk.server.api.ManagedNamespaceWithLifecycle
import org.eclipse.milo.opcua.sdk.server.api.MonitoredItem
import org.eclipse.milo.opcua.sdk.server.api.config.OpcUaServerConfig
import org.eclipse.milo.opcua.sdk.server.nodes.UaFolderNode
import org.eclipse.milo.opcua.sdk.server.nodes.UaNode
import org.eclipse.milo.opcua.sdk.server.nodes.UaVariableNode
import org.eclipse.milo.opcua.sdk.server.nodes.factories.NodeFactory
import org.eclipse.milo.opcua.sdk.server.util.SubscriptionModel
import org.eclipse.milo.opcua.stack.core.Identifiers
import org.eclipse.milo.opcua.stack.core.types.builtin.*
import org.eclipse.milo.opcua.stack.core.types.enumerated.MessageSecurityMode
import org.eclipse.milo.opcua.stack.core.security.SecurityPolicy
import org.eclipse.milo.opcua.stack.core.transport.TransportProfile
import org.eclipse.milo.opcua.stack.server.EndpointConfiguration
import org.springframework.stereotype.Component

@Component
class OpcUaServer {
    private val logger = LoggerFactory.getLogger(javaClass)
    private lateinit var server: OpcUaServer
    private lateinit var namespace: CncNamespace
    private val NAMESPACE_URI = "urn:cnc:opcua:server"

    fun startup() {
        val endpoints = createEndpointConfigurations()
        
        val config = OpcUaServerConfig.builder()
            .setApplicationName(LocalizedText.english("CNC OPC-UA Server"))
            .setApplicationUri("urn:cnc:opcua:server")
            .setProductUri("urn:cnc:opcua:server")
            .setEndpoints(endpoints)
            .build()

        server = OpcUaServer(config)
        namespace = CncNamespace(server)
        
        server.startup().get()
        namespace.startup()
        
        logger.info("OPC UA Server started")
    }

    private fun createEndpointConfigurations(): Set<EndpointConfiguration> {
        return setOf(
            EndpointConfiguration.newBuilder()
                .setBindAddress("0.0.0.0")
                .setHostname("localhost")
                .setPath("/milo")
                .setTransportProfile(TransportProfile.TCP_UASC_UABINARY)
                .setSecurityPolicy(SecurityPolicy.None)
                .setSecurityMode(MessageSecurityMode.None)
                .setBindPort(12686)
                .build()
        )
    }

    inner class CncNamespace(server: OpcUaServer) : ManagedNamespaceWithLifecycle(server, NAMESPACE_URI) {
        private val subscriptionModel = SubscriptionModel(server, this)
        private lateinit var cncFolder: UaFolderNode
        private val variableNodes = mutableMapOf<String, UaVariableNode>()
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

        init {
            // Add lifecycle to server
            getLifecycleManager().addLifecycle(subscriptionModel)
            // Add startup task
            getLifecycleManager().addStartupTask {
                initializeNodes()
                startDataUpdates()
            }
        }

        private fun initializeNodes() {
            // Create CNC folder
            cncFolder = UaFolderNode(
                getNodeContext(),
                newNodeId("CNC_Data"),
                QualifiedName(namespaceIndex, "CNC_Data"),
                LocalizedText.english("CNC_Data")
            )

            getNodeManager().addNode(cncFolder)
            cncFolder.addReference(Reference(
                cncFolder.nodeId,
                Identifiers.Organizes,
                Identifiers.ObjectsFolder.expanded(),
                false
            ))

            // Initialize all nodes
            initializeAxisNodes()
            initializeMachiningNodes()
        }

        private fun initializeAxisNodes() {
            // X Axis Nodes
            listOf(
                "X_ActualPosition", "X_ActualVelocity", "X_ActualAcceleration",
                "X_SetPosition", "X_SetVelocity", "X_SetAcceleration",
                "X_CurrentFeedback", "X_DCBusVoltage", "X_OutputVoltage",
                "X_OutputPower"
            ).forEach { createDoubleVariableNode(it) }
            createIntegerVariableNode("X_OutputCurrent")

            // Y Axis Nodes (same structure as X)
            listOf(
                "Y_ActualPosition", "Y_ActualVelocity", "Y_ActualAcceleration",
                "Y_SetPosition", "Y_SetVelocity", "Y_SetAcceleration",
                "Y_CurrentFeedback", "Y_DCBusVoltage", "Y_OutputVoltage",
                "Y_OutputPower"
            ).forEach { createDoubleVariableNode(it) }
            createIntegerVariableNode("Y_OutputCurrent")

            // Z Axis Nodes (no OutputPower)
            listOf(
                "Z_ActualPosition", "Z_ActualVelocity", "Z_ActualAcceleration",
                "Z_SetPosition", "Z_SetVelocity", "Z_SetAcceleration",
                "Z_CurrentFeedback", "Z_DCBusVoltage", "Z_OutputVoltage"
            ).forEach { createDoubleVariableNode(it) }
            createIntegerVariableNode("Z_OutputCurrent")

            // Spindle Nodes
            listOf(
                "S_ActualPosition", "S_ActualVelocity", "S_ActualAcceleration",
                "S_SetPosition", "S_SetVelocity", "S_SetAcceleration",
                "S_CurrentFeedback", "S_DCBusVoltage", "S_OutputVoltage",
                "S_OutputPower", "S_SystemInertia"
            ).forEach { createDoubleVariableNode(it) }
            createIntegerVariableNode("S_OutputCurrent")
        }

        private fun initializeMachiningNodes() {
            listOf(
                "M_CURRENT_PROGRAM_NUMBER",
                "M_sequence_number",
                "M_CURRENT_FEEDRATE",
                "Machining_Process"
            ).forEach { createStringVariableNode(it) }
        }

        private fun createDoubleVariableNode(name: String) {
            val node = UaVariableNode.builder(getNodeContext())
                .setNodeId(newNodeId(name))
                .setAccessLevel(AccessLevel.READ_WRITE)
                .setUserAccessLevel(AccessLevel.READ_WRITE)
                .setBrowseName(QualifiedName(namespaceIndex, name))
                .setDisplayName(LocalizedText.english(name))
                .setDataType(Identifiers.Double)
                .setTypeDefinition(Identifiers.BaseDataVariableType)
                .build()

            node.value = DataValue(Variant(0.0))
            
            getNodeManager().addNode(node)
            cncFolder.addOrganizes(node)
            variableNodes[name] = node
        }

        private fun createIntegerVariableNode(name: String) {
            val node = UaVariableNode.builder(getNodeContext())
                .setNodeId(newNodeId(name))
                .setAccessLevel(AccessLevel.READ_WRITE)
                .setUserAccessLevel(AccessLevel.READ_WRITE)
                .setBrowseName(QualifiedName(namespaceIndex, name))
                .setDisplayName(LocalizedText.english(name))
                .setDataType(Identifiers.Int32)
                .setTypeDefinition(Identifiers.BaseDataVariableType)
                .build()

            node.value = DataValue(Variant(0))
            
            getNodeManager().addNode(node)
            cncFolder.addOrganizes(node)
            variableNodes[name] = node
        }

        private fun createStringVariableNode(name: String) {
            val node = UaVariableNode.builder(getNodeContext())
                .setNodeId(newNodeId(name))
                .setAccessLevel(AccessLevel.READ_WRITE)
                .setUserAccessLevel(AccessLevel.READ_WRITE)
                .setBrowseName(QualifiedName(namespaceIndex, name))
                .setDisplayName(LocalizedText.english(name))
                .setDataType(Identifiers.String)
                .setTypeDefinition(Identifiers.BaseDataVariableType)
                .build()

            node.value = DataValue(Variant(""))
            
            getNodeManager().addNode(node)
            cncFolder.addOrganizes(node)
            variableNodes[name] = node
        }

        private fun startDataUpdates() {
            fixedRateTimer("DataUpdate", true, 0L, 1000L) {
                try {
                    updateNodesWithNextData()
                } catch (e: Exception) {
                    logger.error("Error updating nodes: ${e.message}")
                }
            }
        }

        private fun updateNodesWithNextData() {
            try {
                val targetFile = File(dataDirectory, targetFiles.random())
                if (!targetFile.exists()) {
                    logger.error("File not found: ${targetFile.absolutePath}")
                    return
                }

                CSVReader(FileReader(targetFile)).use { reader ->
                    // Skip header
                    reader.readNext()

                    // Read random line
                    val lines = reader.readAll()
                    if (lines.isEmpty()) {
                        logger.error("No data in file: ${targetFile.absolutePath}")
                        return
                    }

                    val csvLine = lines.random()

                    // X축 데이터 업데이트
                    updateAxisData("X", csvLine, 0)
                    // Y축 데이터 업데이트
                    updateAxisData("Y", csvLine, 11)
                    // Z축 데이터 업데이트
                    updateAxisData("Z", csvLine, 22)
                    // 스핀들 데이터 업데이트
                    updateSpindleData(csvLine, 32)
                    // 머신 데이터 업데이트
                    updateMachineData(csvLine, 44)
                }
            } catch (e: Exception) {
                logger.error("Error reading CSV: ${e.message}")
            }
        }

        private fun updateAxisData(axis: String, csvLine: Array<String>, startIndex: Int) {
            val nodes = mapOf(
                "${axis}_ActualPosition" to startIndex,
                "${axis}_ActualVelocity" to startIndex + 1,
                "${axis}_ActualAcceleration" to startIndex + 2,
                "${axis}_SetPosition" to startIndex + 3,
                "${axis}_SetVelocity" to startIndex + 4,
                "${axis}_SetAcceleration" to startIndex + 5,
                "${axis}_CurrentFeedback" to startIndex + 6,
                "${axis}_DCBusVoltage" to startIndex + 7,
                "${axis}_OutputCurrent" to startIndex + 8,
                "${axis}_OutputVoltage" to startIndex + 9
            )

            nodes.forEach { (nodeName, index) ->
                try {
                    val value = when (nodeName.endsWith("OutputCurrent")) {
                        true -> csvLine[index].toInt()
                        false -> csvLine[index].toDouble()
                    }
                    variableNodes[nodeName]?.value = DataValue(Variant(value))
                } catch (e: Exception) {
                    logger.error("Error updating $nodeName: ${e.message}")
                }
            }

            // OutputPower는 Z축에는 없음
            if (axis != "Z") {
                try {
                    val value = csvLine[startIndex + 10].toDouble()
                    variableNodes["${axis}_OutputPower"]?.value = DataValue(Variant(value))
                } catch (e: Exception) {
                    logger.error("Error updating ${axis}_OutputPower: ${e.message}")
                }
            }
        }

        private fun updateSpindleData(csvLine: Array<String>, startIndex: Int) {
            val nodes = mapOf(
                "S_ActualPosition" to startIndex,
                "S_ActualVelocity" to startIndex + 1,
                "S_ActualAcceleration" to startIndex + 2,
                "S_SetPosition" to startIndex + 3,
                "S_SetVelocity" to startIndex + 4,
                "S_SetAcceleration" to startIndex + 5,
                "S_CurrentFeedback" to startIndex + 6,
                "S_DCBusVoltage" to startIndex + 7,
                "S_OutputCurrent" to startIndex + 8,
                "S_OutputVoltage" to startIndex + 9,
                "S_OutputPower" to startIndex + 10,
                "S_SystemInertia" to startIndex + 11
            )

            nodes.forEach { (nodeName, index) ->
                try {
                    val value = when (nodeName.endsWith("OutputCurrent")) {
                        true -> csvLine[index].toInt()
                        false -> csvLine[index].toDouble()
                    }
                    variableNodes[nodeName]?.value = DataValue(Variant(value))
                } catch (e: Exception) {
                    logger.error("Error updating $nodeName: ${e.message}")
                }
            }
        }

        private fun updateMachineData(csvLine: Array<String>, startIndex: Int) {
            val nodes = mapOf(
                "M_CURRENT_PROGRAM_NUMBER" to startIndex,
                "M_sequence_number" to startIndex + 1,
                "M_CURRENT_FEEDRATE" to startIndex + 2,
                "Machining_Process" to startIndex + 3
            )

            nodes.forEach { (nodeName, index) ->
                try {
                    variableNodes[nodeName]?.value = DataValue(Variant(csvLine[index]))
                } catch (e: Exception) {
                    logger.error("Error updating $nodeName: ${e.message}")
                }
            }
        }

        override fun onDataItemsCreated(dataItems: List<DataItem>) {
            subscriptionModel.onDataItemsCreated(dataItems)
        }

        override fun onDataItemsModified(dataItems: List<DataItem>) {
            subscriptionModel.onDataItemsModified(dataItems)
        }

        override fun onDataItemsDeleted(dataItems: List<DataItem>) {
            subscriptionModel.onDataItemsDeleted(dataItems)
        }

        override fun onMonitoringModeChanged(monitoredItems: List<MonitoredItem>) {
            subscriptionModel.onMonitoringModeChanged(monitoredItems)
        }
    }

    fun shutdown() {
        if (::server.isInitialized) {
            server.shutdown().get()
        }
    }
}