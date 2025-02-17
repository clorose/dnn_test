// path: frontend/src/pages/graphPage/graphPage.tsx
import { useState, useEffect } from "react";
import {
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Bar,
} from "recharts";
import CustomBar from "./components/CustomBar";
import CustomScatter from "./components/CustomScatter";

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

interface QualityResult {
  passed: boolean;
  ai_prediction: number;
  errors: {
    X: number;
    Y: number;
    Z: number;
    S: number;
  };
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
  quality: QualityResult;
}

const GraphPage = () => {
  const [data, setData] = useState<MachineData | null>(null);
  const [qualityCounts, setQualityCounts] = useState({ pass: 0, fail: 0 });
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [outputCurrentData, setOutputCurrentData] = useState<{
    X: { current: number; passRate: number }[];
    Y: { current: number; passRate: number }[];
  }>({
    X: [],
    Y: []
  });

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch("http://localhost:3000/test");
        if (response.ok) {
          console.log("서버 연결 성공");
          setIsConnected(true);
        }
      } catch (error) {
        console.error("서버 연결 실패:", error);
        setIsConnected(false);
        setError(
          `서버 연결 실패: ${error instanceof Error ? error.message : "알 수 없는 오류"}`
        );
      }
    };
    checkConnection();
  }, []);

  useEffect(() => {
    if (!isConnected) return;
    const fetchData = async () => {
      try {
        const response = await fetch("http://localhost:3000/machine");
        if (!response.ok) {
          throw new Error("데이터 조회 실패");
        }
        const newData: MachineData = await response.json();
        setData(newData);

        const currentPoint = {
          X: {
            current: newData.X.OutputCurrent,
            passRate: 100 - newData.quality.errors.X
          },
          Y: {
            current: newData.Y.OutputCurrent,
            passRate: 100 - newData.quality.errors.Y
          }
        };

        setOutputCurrentData(prev => {
          const newState = {
            X: [...prev.X, currentPoint.X].slice(-20),
            Y: [...prev.Y, currentPoint.Y].slice(-20)
          };
          return newState;
        });

        const isPassed = newData.AI_Predict >= 0.5;

        if (newData.quality) {
          setQualityCounts((prev) => ({
            pass: prev.pass + (isPassed ? 1 : 0),
            fail: prev.fail + (isPassed ? 0 : 1),
          }));
        }
      } catch (error) {
        console.error("데이터 조회 오류:", error);
        setError(
          error instanceof Error ? error.message : "데이터 조회 중 오류 발생"
        );
      }
    };

    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [isConnected]);

  if (!isConnected) {
    return (
      <div className="h-screen w-full flex items-center justify-center">
        <div className="text-red-500">
          서버 연결 실패. 서버 상태를 확인해주세요.
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen w-full flex items-center justify-center">
        <div className="text-red-500">Error: {error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="h-screen w-full flex items-center justify-center">
        <div>Loading...</div>
      </div>
    );
  }

  return (
    <div className="h-screen w-full flex flex-col">
      <h1 className="flex h-12 justify-center items-center bg-sky-200">
        품질 대시보드
      </h1>

      <div className="flex-1 grid grid-cols-2 gap-4 p-4">
        {/* 좌상단: 품질 결과 */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <h2 className="text-xl font-bold">품질 결과</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={[
                { name: "Pass", count: qualityCounts.pass },
                { name: "Fail", count: qualityCounts.fail }
              ]}
              layout="vertical"
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={100} />
              <Tooltip />
              <Bar dataKey="count" barSize={40} shape={<CustomBar />} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 우상단: Current Machine Data */}
        <div className="bg-white shadow-lg rounded-lg p-4 overflow-auto aspect-[4/3]">
          <h2 className="text-xl font-bold">Current Machine Data</h2>
          <table className="min-w-full text-sm mt-4">
            <thead>
              <tr className="bg-gray-50">
                <th className="px-2 py-1 text-left">축</th>
                <th className="px-2 py-1 text-left">파라미터</th>
                <th className="px-2 py-1 text-right">값</th>
              </tr>
            </thead>
            <tbody>
              {["X", "Y", "Z", "S"].map((axis) => {
                const axisData = data[axis as keyof Pick<MachineData, "X" | "Y" | "Z" | "S">];
                if (!axisData) return null;
                return Object.entries(axisData).map(([param, value], idx) => (
                  <tr key={`${axis}-${param}`} className="border-t">
                    {idx === 0 && (
                      <td
                        className="px-2 py-1 font-bold"
                        rowSpan={Object.keys(axisData).length}
                      >
                        {axis}
                      </td>
                    )}
                    <td className="px-2 py-1">{param}</td>
                    <td className="px-2 py-1 text-right">
                      {typeof value === "number" ? value.toFixed(1) : value}
                    </td>
                  </tr>
                ));
              })}
              {data.M &&
                Object.entries(data.M).map(([param, value], idx) => (
                  <tr key={`M-${param}`} className="border-t">
                    {idx === 0 && (
                      <td
                        className="px-2 py-1 font-bold"
                        rowSpan={Object.keys(data.M).length}
                      >
                        M
                      </td>
                    )}
                    <td className="px-2 py-1">{param}</td>
                    <td className="px-2 py-1 text-right">{value}</td>
                  </tr>
                ))}
              <tr className="border-t">
                <td className="px-2 py-1 font-bold">상태</td>
                <td className="px-2 py-1" colSpan={2}>
                  {data.status}
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* 좌하단: X축 OutputCurrent */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <CustomScatter
            title="X축 OutputCurrent"
            data={outputCurrentData.X}
          />
        </div>

        {/* 우하단: Y축 OutputCurrent */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <CustomScatter
            title="Y축 OutputCurrent"
            data={outputCurrentData.Y}
          />
        </div>
      </div>
    </div>
  );
};

export default GraphPage;