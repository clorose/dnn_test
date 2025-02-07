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
import { mockMachineData } from "./mocks/mockData";
import CustomBar from "./components/CustomBar";

const GraphPage = () => {
  const [data, setData] = useState(mockMachineData[0]);
  const [qualityCounts, setQualityCounts] = useState({ pass: 0, fail: 0 });

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      if (index >= mockMachineData.length) index = 0;
      const newData = mockMachineData[index];
      setData(newData);

      // 품질 결과 누적
      setQualityCounts((prev) => ({
        pass: prev.pass + (newData.quality.passed ? 1 : 0),
        fail: prev.fail + (newData.quality.passed ? 0 : 1),
      }));

      index++;
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // 차트 데이터 계산
  const chartData = [
    {
      name: "Pass",
      count: qualityCounts.pass,
    },
    {
      name: "Fail",
      count: qualityCounts.fail,
    },
  ];

  return (
    <div className="h-screen w-full flex flex-col">
      <h1 className="flex h-12 justify-center items-center bg-sky-200">
        Quality Dashboard
      </h1>
      <div className="flex-1 flex p-4 gap-4">
        {/* 품질 결과 차트 */}
        <div className="w-1/2 h-[600px] bg-white shadow-lg rounded-lg">
          <div className="p-4 h-full">
            <h2 className="text-xl font-bold mb-4">Quality Results</h2>
            <div className="h-[calc(100%-2rem)]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={chartData}
                  margin={{ top: 5, right: 30, left: 50, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="name" width={100} />
                  <Tooltip />
                  <Bar dataKey="count" barSize={40} shape={<CustomBar />} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* 현재 머신 데이터 테이블 */}
        <div className="w-1/2 h-[600px] bg-white shadow-lg rounded-lg">
          <div className="p-4 h-full overflow-auto">
            <h2 className="text-xl font-bold mb-4">Current Machine Data</h2>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-2 py-1 text-left">Axis</th>
                  <th className="px-2 py-1 text-left">Parameter</th>
                  <th className="px-2 py-1 text-right">Value</th>
                </tr>
              </thead>
              <tbody>
                {/* 축 데이터 (X, Y, Z, S) */}
                {(["X", "Y", "Z", "S"] as const).map((axis) =>
                  Object.entries(data[axis]).map(([param, value], idx) => (
                    <tr key={`${axis}-${param}`} className="border-t">
                      {idx === 0 ? (
                        <td
                          className="px-2 py-1 font-bold"
                          rowSpan={Object.keys(data[axis]).length}
                        >
                          {axis}
                        </td>
                      ) : null}
                      <td className="px-2 py-1">{param}</td>
                      <td className="px-2 py-1 text-right">
                        {typeof value === "number" ? value.toFixed(3) : value}
                      </td>
                    </tr>
                  ))
                )}
                {/* M 데이터 */}
                {Object.entries(data.M).map(([param, value], idx) => (
                  <tr key={`M-${param}`} className="border-t">
                    {idx === 0 ? (
                      <td
                        className="px-2 py-1 font-bold"
                        rowSpan={Object.keys(data.M).length}
                      >
                        M
                      </td>
                    ) : null}
                    <td className="px-2 py-1">{param}</td>
                    <td className="px-2 py-1 text-right">{value}</td>
                  </tr>
                ))}
                {/* 상태 정보 */}
                <tr className="border-t">
                  <td className="px-2 py-1 font-bold">Status</td>
                  <td className="px-2 py-1" colSpan={2}>
                    {data.status}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GraphPage;
