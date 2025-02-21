// path: frontend/src/pages/graphPage/types/chartTypes.ts
export interface ScatterData {
  current: number;
  passRate: number;
}

export interface TrackingData {
  timestamp: string;
  setPoint: number;
  actualPoint: number;
  passed: boolean;
}

export interface TrackingChartProps {
  title: string;
  data: TrackingData[];
  yAxisLabel: string;
}

export interface FailRange {
  start: number;
  end: number;
}