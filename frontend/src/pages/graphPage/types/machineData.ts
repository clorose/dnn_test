export interface AxisData {
  ActualPosition: number;
  ActualVelocity: number;
  ActualAcceleration: number;
  SetPosition: number;
  SetVelocity: number;
  SetAcceleration: number;
  CurrentFeedback: number;
  DCBusVoltage: number;
  OutputCurrent: number;
  OutputVoltage: number;
  OutputPower: number;
}

export interface SAxisData extends AxisData {
  SystemInertia: number;
}

export interface QualityResult {
  passed: boolean | null;
  ai_prediction: number | null;
}

export interface MachineData {
  timestamp: string;
  X: AxisData;
  Y: AxisData;
  Z: AxisData;
  S: SAxisData;
  M: {
    CURRENT_PROGRAM_NUMBER: number;
    sequence_number: number;
    CURRENT_FEEDRATE: number;
  };
  status: string;
  quality: QualityResult;
}

export interface ScatterData {
  current: number;
  passRate: number;
}