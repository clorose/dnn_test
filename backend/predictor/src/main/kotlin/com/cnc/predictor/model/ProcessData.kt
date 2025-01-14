// path: predictor/src/main/kotlin/com/cnc/predictor/model/ProcessData.kt
package com.cnc.predictor.model

data class ProcessData(
    // X축의 실제 위치 (단위: mm)
    val X_ActualPosition: Double,
    // X축의 실제 속도 (단위: mm/s)
    val X_ActualVelocity: Double,
    // X축의 실제 가속도 (단위: mm/s²)
    val X_ActualAcceleration: Double,
    // X축의 설정 목표 위치 (단위: mm)
    val X_SetPosition: Double,
    // X축의 설정 목표 속도 (단위: mm/s)
    val X_SetVelocity: Double,
    // X축의 설정 목표 가속도 (단위: mm/s²)
    val X_SetAcceleration: Double,
    // X축의 전류 피드백 값 (단위: A)
    val X_CurrentFeedback: Double,
    // X축의 DC 버스 전압 (단위: V)
    val X_DCBusVoltage: Double,
    // X축의 출력 전류 (단위: mA)
    val X_OutputCurrent: Int,
    // X축의 출력 전압 (단위: V)
    val X_OutputVoltage: Double,
    // X축의 출력 전력 (단위: W)
    val X_OutputPower: Double,

    // Y축의 실제 위치 (단위: mm)
    val Y_ActualPosition: Double,
    // Y축의 실제 속도 (단위: mm/s)
    val Y_ActualVelocity: Double,
    // Y축의 실제 가속도 (단위: mm/s²)
    val Y_ActualAcceleration: Double,
    // Y축의 설정 목표 위치 (단위: mm)
    val Y_SetPosition: Double,
    // Y축의 설정 목표 속도 (단위: mm/s)
    val Y_SetVelocity: Double,
    // Y축의 설정 목표 가속도 (단위: mm/s²)
    val Y_SetAcceleration: Double,
    // Y축의 전류 피드백 값 (단위: A)
    val Y_CurrentFeedback: Double,
    // Y축의 DC 버스 전압 (단위: V)
    val Y_DCBusVoltage: Double,
    // Y축의 출력 전류 (단위: mA)
    val Y_OutputCurrent: Int,
    // Y축의 출력 전압 (단위: V)
    val Y_OutputVoltage: Double,
    // Y축의 출력 전력 (단위: W)
    val Y_OutputPower: Double,

    // Z축의 실제 위치 (단위: mm)
    val Z_ActualPosition: Double,
    // Z축의 실제 속도 (단위: mm/s)
    val Z_ActualVelocity: Double,
    // Z축의 실제 가속도 (단위: mm/s²)
    val Z_ActualAcceleration: Double,
    // Z축의 설정 목표 위치 (단위: mm)
    val Z_SetPosition: Double,
    // Z축의 설정 목표 속도 (단위: mm/s)
    val Z_SetVelocity: Double,
    // Z축의 설정 목표 가속도 (단위: mm/s²)
    val Z_SetAcceleration: Double,
    // Z축의 전류 피드백 값 (단위: A)
    val Z_CurrentFeedback: Double,
    // Z축의 DC 버스 전압 (단위: V)
    val Z_DCBusVoltage: Double,
    // Z축의 출력 전류 (단위: mA)
    val Z_OutputCurrent: Int,
    // Z축의 출력 전압 (단위: V)
    val Z_OutputVoltage: Double,

    // 스핀들의 실제 위치 (단위: degree)
    val S_ActualPosition: Double,
    // 스핀들의 실제 속도 (단위: rpm)
    val S_ActualVelocity: Double,
    // 스핀들의 실제 가속도 (단위: rpm/s)
    val S_ActualAcceleration: Double,
    // 스핀들의 설정 목표 위치 (단위: degree)
    val S_SetPosition: Double,
    // 스핀들의 설정 목표 속도 (단위: rpm)
    val S_SetVelocity: Double,
    // 스핀들의 설정 목표 가속도 (단위: rpm/s)
    val S_SetAcceleration: Double,
    // 스핀들의 전류 피드백 값 (단위: A)
    val S_CurrentFeedback: Double,
    // 스핀들의 DC 버스 전압 (단위: V)
    val S_DCBusVoltage: Double,
    // 스핀들의 출력 전류 (단위: mA)
    val S_OutputCurrent: Int,
    // 스핀들의 출력 전압 (단위: V)
    val S_OutputVoltage: Double,
    // 스핀들의 출력 전력 (단위: W)
    val S_OutputPower: Double,
    // 스핀들 시스템의 관성 (단위: kg⋅m²)
    val S_SystemInertia: Double,

    // 현재 실행 중인 프로그램 번호
    val M_CURRENT_PROGRAM_NUMBER: String,
    // 현재 실행 중인 시퀀스 번호
    val M_sequence_number: String,
    // 현재 이송 속도 (단위: mm/min)
    val M_CURRENT_FEEDRATE: String,
    // 현재 가공 공정 단계
    val Machining_Process: String,
)