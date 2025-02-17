// path: frontend/src/pages/graphPage/mocks/mockData.ts
interface AxisData {
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

interface SAxisData extends AxisData {
  SystemInertia: number;
}

interface MachineData {
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
}

interface QualityResult {
  passed: boolean;
  errors: {
    X: number;
    Y: number;
    Z: number;
    S: number;
  };
}

// 시간에 따른 데이터 포인트 생성을 위한 헬퍼 함수
const generateTimeSeriesData = (length: number, baseValue: number, amplitude: number, noise: number = 0.2) => {
  return Array.from({ length }, (_, i) => {
    return baseValue + Math.sin(i * 0.1) * amplitude + (Math.random() - 0.5) * amplitude * noise;
  });
};

// 품질 체크 함수
const checkQuality = (machineData: MachineData): QualityResult => {
  // 각 축의 Velocity 오차 계산
  const xVelError = Math.abs(machineData.X.ActualVelocity - machineData.X.SetVelocity);
  const yVelError = Math.abs(machineData.Y.ActualVelocity - machineData.Y.SetVelocity);
  const zVelError = Math.abs(machineData.Z.ActualVelocity - machineData.Z.SetVelocity);
  const sVelError = Math.abs(machineData.S.ActualVelocity - machineData.S.SetVelocity);

  // 허용 오차 범위 (각 축의 IQR 값 기반)
  const VELOCITY_THRESHOLDS = {
    X: 9.85 * 0.5,  // IQR의 50%
    Y: 3.09 * 0.5,
    Z: 0.5 * 0.5,
    S: 0.5 * 0.5
  };

  // 각 축별 판정
  const isXPass = xVelError <= VELOCITY_THRESHOLDS.X;
  const isYPass = yVelError <= VELOCITY_THRESHOLDS.Y;
  const isZPass = zVelError <= VELOCITY_THRESHOLDS.Z;
  const isSPass = sVelError <= VELOCITY_THRESHOLDS.S;

  // 최종 판정 (모든 축이 Pass여야 최종 Pass)
  return {
    passed: isXPass && isYPass && isZPass && isSPass,
    errors: {
      X: xVelError,
      Y: yVelError,
      Z: zVelError,
      S: sVelError
    }
  };
};

// 실제 통계 기반 Mock 데이터 생성
export const mockMachineData = Array.from({ length: 50 }, (_, index) => {
  const timeStamp = new Date(2024, 1, 1, 0, index).toISOString();

  const machineState: MachineData = {
    timestamp: timeStamp,
    X: {
      ActualPosition: generateTimeSeriesData(1, 155, 8)[0],
      ActualVelocity: generateTimeSeriesData(1, 4, 10)[0],
      ActualAcceleration: generateTimeSeriesData(1, 4, 75)[0],
      SetPosition: generateTimeSeriesData(1, 155, 4)[0],
      SetVelocity: generateTimeSeriesData(1, 4, 5)[0],
      SetAcceleration: generateTimeSeriesData(1, 4, 35)[0],
      CurrentFeedback: generateTimeSeriesData(1, 0, 5)[0],
      DCBusVoltage: generateTimeSeriesData(1, 0.08, 0.03)[0],
      OutputCurrent: generateTimeSeriesData(1, 326, 0.5)[0],
      OutputVoltage: generateTimeSeriesData(1, 10, 6)[0],
      OutputPower: generateTimeSeriesData(1, 0.001, 0.001)[0]
    },
    Y: {
      ActualPosition: generateTimeSeriesData(1, 93.5, 25)[0],
      ActualVelocity: generateTimeSeriesData(1, 4, 3)[0],
      ActualAcceleration: generateTimeSeriesData(1, 4, 50)[0],
      SetPosition: generateTimeSeriesData(1, 93.5, 12.5)[0],
      SetVelocity: generateTimeSeriesData(1, 4, 1.5)[0],
      SetAcceleration: generateTimeSeriesData(1, 4, 25)[0],
      CurrentFeedback: generateTimeSeriesData(1, 0, 5)[0],
      DCBusVoltage: generateTimeSeriesData(1, 0.08, 0.04)[0],
      OutputCurrent: generateTimeSeriesData(1, 324, 0.6)[0],
      OutputVoltage: generateTimeSeriesData(1, 9, 8)[0],
      OutputPower: generateTimeSeriesData(1, 0.001, 0.002)[0]
    },
    Z: {
      ActualPosition: generateTimeSeriesData(1, 32.5, 2)[0],
      ActualVelocity: generateTimeSeriesData(1, 4, 0.5)[0],
      ActualAcceleration: generateTimeSeriesData(1, 4, 0.5)[0],
      SetPosition: generateTimeSeriesData(1, 32.5, 1)[0],
      SetVelocity: generateTimeSeriesData(1, 4, 0.25)[0],
      SetAcceleration: generateTimeSeriesData(1, 4, 0.25)[0],
      CurrentFeedback: generateTimeSeriesData(1, 0, 0.1)[0],
      DCBusVoltage: generateTimeSeriesData(1, 0, 0.1)[0],
      OutputCurrent: generateTimeSeriesData(1, 0, 0.1)[0],
      OutputVoltage: generateTimeSeriesData(1, 0, 0.1)[0],
      OutputPower: generateTimeSeriesData(1, 0, 0)[0]
    },
    S: {
      ActualPosition: generateTimeSeriesData(1, 35, 25)[0],
      ActualVelocity: generateTimeSeriesData(1, 57, 0.5)[0],
      ActualAcceleration: generateTimeSeriesData(1, 4, 6)[0],
      SetPosition: generateTimeSeriesData(1, 35, 25)[0],
      SetVelocity: generateTimeSeriesData(1, 57, 0.25)[0],
      SetAcceleration: generateTimeSeriesData(1, 4, 0.25)[0],
      CurrentFeedback: generateTimeSeriesData(1, 20, 5)[0],
      DCBusVoltage: generateTimeSeriesData(1, 0.9, 0.2)[0],
      OutputCurrent: generateTimeSeriesData(1, 320, 2.5)[0],
      OutputVoltage: generateTimeSeriesData(1, 114, 20)[0],
      OutputPower: generateTimeSeriesData(1, 0.17, 0.03)[0],
      SystemInertia: 16
    },
    M: {
      CURRENT_PROGRAM_NUMBER: 1,
      sequence_number: Math.floor(Math.random() * 132),
      CURRENT_FEEDRATE: Math.floor(Math.random() * 44) + 6
    },
    status: ['Starting', 'Prep', 'Layer 1 Up', 'Layer 1 Down', 'Layer 2 Up', 'Layer 2 Down', 'Layer 3 Up', 'Layer 3 Down', 'Repositioning', 'end'][Math.floor(Math.random() * 10)]
  };

  // 품질 체크 결과 추가
  const qualityCheck = checkQuality(machineState);
  return {
    ...machineState,
    quality: qualityCheck
  };
});