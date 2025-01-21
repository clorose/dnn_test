// path: /Users/gohan/Develop/dnn_test/backend/opcua/src/main/kotlin/com/cnc/opcua/dto/OpcUaDto.kt
package com.cnc.opcua.dto

data class ProcessDataDto(
    val x_ActualPosition: Double,
    val x_ActualVelocity: Double,
    val x_ActualAcceleration: Double,
    val x_SetPosition: Double,
    val x_SetVelocity: Double,
    val x_SetAcceleration: Double,
    val x_CurrentFeedback: Double,
    val x_DCBusVoltage: Double,
    val x_OutputCurrent: Int,
    val x_OutputVoltage: Double,
    val x_OutputPower: Double,
    
    val y_ActualPosition: Double,
    val y_ActualVelocity: Double,
    val y_ActualAcceleration: Double,
    val y_SetPosition: Double,
    val y_SetVelocity: Double,
    val y_SetAcceleration: Double,
    val y_CurrentFeedback: Double,
    val y_DCBusVoltage: Double,
    val y_OutputCurrent: Int,
    val y_OutputVoltage: Double,
    val y_OutputPower: Double,
    
    val z_ActualPosition: Double,
    val z_ActualVelocity: Double,
    val z_ActualAcceleration: Double,
    val z_SetPosition: Double,
    val z_SetVelocity: Double,
    val z_SetAcceleration: Double,
    val z_CurrentFeedback: Double,
    val z_DCBusVoltage: Double,
    val z_OutputCurrent: Int,
    val z_OutputVoltage: Double,
    
    val s_ActualPosition: Double,
    val s_ActualVelocity: Double,
    val s_ActualAcceleration: Double,
    val s_SetPosition: Double,
    val s_SetVelocity: Double,
    val s_SetAcceleration: Double,
    val s_CurrentFeedback: Double,
    val s_DCBusVoltage: Double,
    val s_OutputCurrent: Int,
    val s_OutputVoltage: Double,
    val s_OutputPower: Double,
    val s_SystemInertia: Double,
    
    val m_CURRENT_PROGRAM_NUMBER: String,
    val m_sequence_number: String,
    val m_CURRENT_FEEDRATE: String,
    val machining_Process: String
)