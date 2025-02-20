// path: frontend/src/types/process.ts
export interface ProcessData {
  // X축 데이터
  X_ActualPosition: number;
  X_ActualVelocity: number;
  X_ActualAcceleration: number;
  X_SetPosition: number;
  X_SetVelocity: number;
  X_SetAcceleration: number;
  X_CurrentFeedback: number;
  X_DCBusVoltage: number;
  X_OutputCurrent: number;
  X_OutputVoltage: number;
  X_OutputPower: number;

  // Y축 데이터
  Y_ActualPosition: number;
  Y_ActualVelocity: number;
  Y_ActualAcceleration: number;
  Y_SetPosition: number;
  Y_SetVelocity: number;
  Y_SetAcceleration: number;
  Y_CurrentFeedback: number;
  Y_DCBusVoltage: number;
  Y_OutputCurrent: number;
  Y_OutputVoltage: number;
  Y_OutputPower: number;

  // Z축 데이터
  Z_ActualPosition: number;
  Z_ActualVelocity: number;
  Z_ActualAcceleration: number;
  Z_SetPosition: number;
  Z_SetVelocity: number;
  Z_SetAcceleration: number;
  Z_CurrentFeedback: number;
  Z_DCBusVoltage: number;
  Z_OutputCurrent: number;
  Z_OutputVoltage: number;

  // 스핀들 데이터
  S_ActualPosition: number;
  S_ActualVelocity: number;
  S_ActualAcceleration: number;
  S_SetPosition: number;
  S_SetVelocity: number;
  S_SetAcceleration: number;
  S_CurrentFeedback: number;
  S_DCBusVoltage: number;
  S_OutputCurrent: number;
  S_OutputVoltage: number;
  S_OutputPower: number;
  S_SystemInertia: number;

  // 기계 정보
  M_CURRENT_PROGRAM_NUMBER: string;
  M_sequence_number: string;
  M_CURRENT_FEEDRATE: string;
  Machining_Process: string;

  // 프론트엔드 전용 필드
  timestamp: string;
  result?: string;
  id: string;
}