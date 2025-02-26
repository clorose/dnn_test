// path: frontend/src/pages/dashboardPage/dashboardPage.tsx
import { useState, useEffect } from "react";
import { useMachineData } from "../graphPage/hooks/useMachineData";
import StatCard from "../../components/StatCard";
import AxisDetailsCard from "../../components/AxisDetailsCard";
import QualityTrendChart from "../../components/QualityTrendChart";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  ReferenceLine
} from "recharts";

interface AxisHistoryData {
  time: string;
  current: number;
  voltage: number;
  power: number;
}

interface QualityHistoryData {
  timestamp: string;
  prediction: number;
  passed: boolean;
}

interface AxisHistory {
  X: AxisHistoryData[];
  Y: AxisHistoryData[];
}

// Scatter 차트에 사용될 데이터 형식 정의
interface ScatterData {
  current: number;
  passRate: number;
}

// 축 OutputCurrent 데이터를 위한 인터페이스
interface AxisCurrentData {
  name: string;
  X: number;
  Y: number;
  Z?: number;
}

const DashboardPage = () => {
  const { data, qualityCounts, error, isConnected, outputCurrentData } = useMachineData();

  const [axisHistory, setAxisHistory] = useState<AxisHistory>({ X: [], Y: [] });
  const [qualityHistory, setQualityHistory] = useState<QualityHistoryData[]>([]);
  const [axisCurrentHistory, setAxisCurrentHistory] = useState<AxisCurrentData[]>([]);

  useEffect(() => {
    if (data) {
      const timestamp = new Date().toLocaleTimeString();

      setAxisHistory((prev) => ({
        X: [...prev.X.slice(-20), { time: timestamp, current: data.X.OutputCurrent, voltage: data.X.OutputVoltage, power: data.X.OutputPower }],
        Y: [...prev.Y.slice(-20), { time: timestamp, current: data.Y.OutputCurrent, voltage: data.Y.OutputVoltage, power: data.Y.OutputPower }]
      }));

      setQualityHistory((prev) => [...prev.slice(-50), { timestamp, prediction: data.quality.ai_prediction ?? 0, passed: data.quality.passed ?? false }]);

      setAxisCurrentHistory((prev) => {
        const newEntry = { name: timestamp, X: data.X.OutputCurrent, Y: data.Y.OutputCurrent, Z: data.Z?.OutputCurrent };
        return [...prev, newEntry].slice(-20);
      });
    }
  }, [data]);

  if (!isConnected || error || !data) {
    return <div className="flex items-center justify-center h-screen text-xl">{error ?? "Connecting..."}</div>;
  }

  const totalCount = qualityCounts.pass + qualityCounts.fail;
  const passRate = totalCount > 0 ? (qualityCounts.pass / totalCount) * 100 : 0;
  const aiPrediction = data.quality.ai_prediction ?? 0;

  const getAxisScatterData = (axis: 'X' | 'Y'): ScatterData[] => outputCurrentData?.[axis] || [];

  return (
    <div className="min-h-screen bg-gray-100 p-3">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-2 mb-4">
        <StatCard title="현재 공정" value={data.status} />
        <StatCard title="품질 예측" value={`${(aiPrediction * 100).toFixed(1)}%`} />
        <StatCard title="합격률" value={`${passRate.toFixed(1)}%`} />
        <StatCard title="시퀀스 번호" value={data.M.sequence_number} />

        {/* 공정 상태 */}
        <div className="bg-white rounded-lg p-4 shadow flex flex-col justify-between">
          <h3 className="text-lg font-bold mb-2">공정 상태</h3>
          <div className="text-sm space-y-2">
            <div className="flex justify-between">
              <span>프로그램 번호</span>
              <span className="font-semibold">{data.M.CURRENT_PROGRAM_NUMBER}</span>
            </div>
            <div className="flex justify-between">
              <span>현재 피드레이트</span>
              <span className="font-semibold">{data.M.CURRENT_FEEDRATE}</span>
            </div>
            <div className="flex justify-between">
              <span>시퀀스 번호</span>
              <span className="font-semibold">{data.M.sequence_number}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* 품질 트렌드 차트 */}
        <div className="col-span-12 lg:col-span-6">
          <QualityTrendChart data={qualityHistory} />
        </div>

        {/* 축 OutputCurrent LineChart */}
        <div className="col-span-12 lg:col-span-6 bg-white rounded-lg p-4 shadow">
          <h3 className="text-lg font-semibold mb-4">각 축 OutputCurrent 추이</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={axisCurrentHistory} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="X" stroke="#8884d8" activeDot={{ r: 8 }} name="X축 Current" />
                <Line type="monotone" dataKey="Y" stroke="#82ca9d" name="Y축 Current" />
                {data.Z && <Line type="monotone" dataKey="Z" stroke="#ff7300" name="Z축 Current" />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 하단 3개 정렬 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-3">
        {/* 축별 OutputCurrent - 합격률 차트 */}
        <div className="bg-white rounded-lg p-4 shadow h-full">
          <h3 className="text-lg font-semibold mb-4">축별 OutputCurrent - 합격률</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="current" name="Current" unit="A" domain={["dataMin", "dataMax"]} />
                <YAxis type="number" dataKey="passRate" name="Pass Rate" unit="%" domain={[0, 100]} />
                <ReferenceLine y={80} stroke="red" strokeDasharray="5 5" label="합격 기준" />
                <Tooltip />
                <Legend verticalAlign="top" height={36} />
                <Scatter name="X축" data={getAxisScatterData('X')} fill="#8884d8" shape="circle" />
                <Scatter name="Y축" data={getAxisScatterData('Y')} fill="#82ca9d" shape="circle" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* X축 상태 카드 */}
        <AxisDetailsCard title="X축 상태" current={data.X.OutputCurrent} voltage={data.X.OutputVoltage} power={data.X.OutputPower} data={axisHistory.X} />

        {/* Y축 상태 카드 */}
        <AxisDetailsCard title="Y축 상태" current={data.Y.OutputCurrent} voltage={data.Y.OutputVoltage} power={data.Y.OutputPower} data={axisHistory.Y} />
      </div>
    </div>
  );
};

export default DashboardPage;
