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

const DashboardPage = () => {
  const { data, qualityCounts, error, isConnected } = useMachineData();

  const [axisHistory, setAxisHistory] = useState<AxisHistory>({
    X: [],
    Y: [],
  });

  const [qualityHistory, setQualityHistory] = useState<QualityHistoryData[]>(
    []
  );

  useEffect(() => {
    if (data) {
      const timestamp = new Date().toLocaleTimeString();

      setAxisHistory((prev) => ({
        X: [
          ...prev.X.slice(-20),
          {
            time: timestamp,
            current: data.X.OutputCurrent,
            voltage: data.X.OutputVoltage,
            power: data.X.OutputPower,
          },
        ],
        Y: [
          ...prev.Y.slice(-20),
          {
            time: timestamp,
            current: data.Y.OutputCurrent,
            voltage: data.Y.OutputVoltage,
            power: data.Y.OutputPower,
          },
        ],
      }));

      setQualityHistory((prev) => [
        ...prev.slice(-50),
        {
          timestamp,
          prediction: data.quality.ai_prediction ?? 0,
          passed: data.quality.passed ?? false,
        },
      ]);
    }
  }, [data]);

  if (!isConnected || error || !data) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-xl">{error ?? "Connecting..."}</div>
      </div>
    );
  }

  const totalCount = qualityCounts.pass + qualityCounts.fail;
  const passRate = totalCount > 0 ? (qualityCounts.pass / totalCount) * 100 : 0;
  const aiPrediction = data.quality.ai_prediction ?? 0;

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Machine Dashboard</h1>
        <div className="space-x-2">
          <span
            className={`px-3 py-1 rounded ${isConnected ? "bg-green-500" : "bg-red-500"
              } text-white`}
          >
            {isConnected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <StatCard title="현재 공정" value={data.status} />
        <StatCard
          title="품질 예측"
          value={`${(aiPrediction * 100).toFixed(1)}%`}
        />
        <StatCard title="합격률" value={`${passRate.toFixed(1)}%`} />
        <StatCard title="시퀀스 번호" value={data.M.sequence_number} />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Quality Trend */}
        <div className="col-span-12 lg:col-span-6">
          <QualityTrendChart data={qualityHistory} />
        </div>

        {/* Position Error */}
        <div className="col-span-12 lg:col-span-6">
          <div className="bg-white rounded-lg p-4 shadow">
            <h3 className="text-lg font-semibold mb-4">Position Error</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                  <CartesianGrid />
                  <XAxis
                    type="number"
                    dataKey="x"
                    name="SetPoint"
                    unit="mm"
                    domain={["auto", "auto"]}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    name="ActualPoint"
                    unit="mm"
                    domain={["auto", "auto"]}
                  />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                  <Scatter
                    name="X축"
                    data={[
                      {
                        x: data.X.SetPosition,
                        y: data.X.ActualPosition,
                      },
                    ]}
                    fill="#8884d8"
                  />
                  <Scatter
                    name="Y축"
                    data={[
                      {
                        x: data.Y.SetPosition,
                        y: data.Y.ActualPosition,
                      },
                    ]}
                    fill="#82ca9d"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Axis Details */}
        <div className="col-span-12 lg:col-span-4">
          <AxisDetailsCard
            title="X축 상태"
            current={data.X.OutputCurrent}
            voltage={data.X.OutputVoltage}
            power={data.X.OutputPower}
            data={axisHistory.X}
          />
        </div>
        <div className="col-span-12 lg:col-span-4">
          <AxisDetailsCard
            title="Y축 상태"
            current={data.Y.OutputCurrent}
            voltage={data.Y.OutputVoltage}
            power={data.Y.OutputPower}
            data={axisHistory.Y}
          />
        </div>

        {/* Process State */}
        <div className="col-span-12 lg:col-span-4">
          <div className="bg-white rounded-lg p-4 shadow">
            <h3 className="text-lg font-semibold mb-4">공정 상태</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span>프로그램 번호</span>
                <span className="font-bold">
                  {data.M.CURRENT_PROGRAM_NUMBER}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span>현재 피드레이트</span>
                <span className="font-bold">{data.M.CURRENT_FEEDRATE}</span>
              </div>
              <div className="flex justify-between items-center">
                <span>시퀀스 번호</span>
                <span className="font-bold">{data.M.sequence_number}</span>
              </div>
              <div className="mt-4">
                <div className="text-sm text-gray-500">공정 진행</div>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full"
                    style={{ width: `${(data.mapped_process / 9) * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
