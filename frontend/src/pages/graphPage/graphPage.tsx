// path: frontend/src/pages/graphPage/graphPage.tsx
import { useMachineData } from "./hooks/useMachineData";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";


import { MachineData } from "./types/machineData";
import CustomBar from "./components/CustomBar";
import CustomScatter from "./components/CustomScatter";
import TrackingChart from "./components/TrackingChart";

const GraphPage = () => {
  const {
    data,
    qualityCounts,
    error,
    isConnected,
    outputCurrentData,
    trackingData,
    powerCurrentData,
  } = useMachineData();

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
      <div className="flex-1 grid grid-cols-2 grid-rows-4 gap-4 p-4">
        {/* 좌상단: 품질 결과 */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <h2 className="text-xl font-bold">품질 결과</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={[
                { name: "Pass", count: qualityCounts.pass },
                { name: "Fail", count: qualityCounts.fail },
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
                const axisData =
                  data[axis as keyof Pick<MachineData, "X" | "Y" | "Z" | "S">];
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
            </tbody>
          </table>
        </div>

        {/* 좌중단: X축 OutputCurrent */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <CustomScatter
            title="X축 OutputCurrent"
            data={outputCurrentData.X}
            xLabel="Output Current"
            yLabel="합격률"
          />
        </div>

        {/* 우중단: Y축 OutputCurrent */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <CustomScatter
            title="Y축 OutputCurrent"
            data={outputCurrentData.Y}
            xLabel="Output Current"
            yLabel="합격률"
          />
        </div>

        {/* 좌하단: X축 Position Tracking */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <TrackingChart
            title="X축 Position Tracking"
            data={trackingData.XPosition}
            yAxisLabel="Position"
          />
        </div>

        {/* 우하단: Y축 Position Tracking */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <TrackingChart
            title="Y축 Position Tracking"
            data={trackingData.YPosition}
            yAxisLabel="Position"
          />
        </div>

        {/* 좌하단: X축 Power vs Feedback */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <CustomScatter
            title="X축 Power-Feedback 관계"
            data={powerCurrentData.X}
            xLabel="Output Power"
            yLabel="Current Feedback"
          />
        </div>

        {/* 우하단: Y축 Power vs Feedback */}
        <div className="bg-white shadow-lg rounded-lg p-4 aspect-[4/3]">
          <CustomScatter
            title="Y축 Power-Feedback 관계"
            data={powerCurrentData.Y}
            xLabel="Output Power"
            yLabel="Current Feedback"
          />
        </div>
      </div>
    </div>
  );
};

export default GraphPage;
