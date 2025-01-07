import { useState, useEffect } from "react";
import { ProcessData } from "../../types/process";

const MainPage = () => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [processData, setProcessData] = useState<ProcessData[]>([]);

  const determineResult = (data: ProcessData): string => {
    const xPosError = Math.abs(
      parseFloat(data.X_ActualPosition) - parseFloat(data.X_SetPosition)
    );
    const yPosError = Math.abs(
      parseFloat(data.Y_ActualPosition) - parseFloat(data.Y_SetPosition)
    );
    const zPosError = Math.abs(
      parseFloat(data.Z_ActualPosition) - parseFloat(data.Z_SetPosition)
    );
    const xCurrentError = Math.abs(parseFloat(data.X_CurrentFeedback));
    const yCurrentError = Math.abs(parseFloat(data.Y_CurrentFeedback));

    if (xPosError > 5 || yPosError > 5 || zPosError > 5) return "불합격";
    if (xCurrentError > 15 || yCurrentError > 15) return "불합격";
    return "합격";
  };

  useEffect(() => {
    let interval: NodeJS.Timeout | undefined;

    if (isConnected) {
      interval = setInterval(() => {
        const newData: ProcessData = {
          X_ActualPosition: (200 + Math.random() * 4 - 2).toFixed(1),
          X_ActualVelocity: (-14 + Math.random() * 2).toFixed(1),
          X_ActualAcceleration: (Math.random() * 100 - 50).toFixed(1),
          X_SetPosition: (200).toFixed(1),
          X_SetVelocity: (-13.9).toFixed(1),
          X_SetAcceleration: (4).toFixed(1),
          X_CurrentFeedback: (-8 + Math.random() * 4).toFixed(2),
          X_DCBusVoltage: (0.12 + Math.random() * 0.02).toFixed(3),
          X_OutputCurrent: (328).toFixed(0),
          X_OutputVoltage: (30 + Math.random() * 2).toFixed(1),
          Y_ActualPosition: (160 + Math.random() * 4 - 2).toFixed(1),
          Y_ActualVelocity: (-28 + Math.random() * 1).toFixed(1),
          Y_ActualAcceleration: (Math.random() * 100 - 50).toFixed(1),
          Y_SetPosition: (160).toFixed(1),
          Y_SetVelocity: (-28.3).toFixed(1),
          Y_SetAcceleration: (4).toFixed(1),
          Y_CurrentFeedback: (-10 + Math.random() * 4).toFixed(2),
          Y_DCBusVoltage: (0.17 + Math.random() * 0.02).toFixed(3),
          Y_OutputCurrent: (326).toFixed(0),
          Y_OutputVoltage: (48 + Math.random() * 3).toFixed(1),
          Z_ActualPosition: (120 + Math.random() * 4 - 2).toFixed(1),
          Z_ActualVelocity: (-29.7 + Math.random() * 0.4).toFixed(1),
          Z_ActualAcceleration: (Math.random() * 100 - 50).toFixed(1),
          Z_SetPosition: (120).toFixed(1),
          Z_SetVelocity: (-29.7).toFixed(1),
          Z_SetAcceleration: (4).toFixed(1),
          Z_CurrentFeedback: (0).toFixed(1),
          Z_DCBusVoltage: (0).toFixed(1),
          Z_OutputCurrent: (0).toFixed(1),
          Z_OutputVoltage: (0).toFixed(1),
          M_CURRENT_PROGRAM_NUMBER: "16",
          M_sequence_number: "1",
          M_CURRENT_FEEDRATE: "50",
          Machining_Process: "Layer 1 Up",
          timestamp: new Date().toISOString(),
          id: crypto.randomUUID(),
        };

        newData.result = determineResult(newData);
        setProcessData((prev) => [...prev, newData].slice(-10));
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isConnected]);

  const formatColumnHeader = (col: string): string => {
    return col
      .split("_")
      .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
      .join(" ");
  };

  return (
    <div className="flex items-center justify-center w-screen h-screen bg-gray-100">
      <div className="w-[90vw] h-[95vh] bg-white rounded-lg shadow-lg overflow-hidden">
        {/* Header Section */}
        <div className="h-16 px-6 bg-white border-b flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-800">
            공정 데이터 모니터링
          </h1>
          <button
            className={`px-6 py-2 rounded-lg font-medium transition-colors shadow-sm
            ${
              isConnected
                ? "bg-red-500 hover:bg-red-600 text-white"
                : "bg-green-500 hover:bg-green-600 text-white"
            }`}
            onClick={() => setIsConnected(!isConnected)}
          >
            {isConnected ? "통신 중지" : "통신 시작"}
          </button>
        </div>

        {/* Table Section with Card-like styling */}
        <div className="p-6 h-[calc(100%-4rem)]">
          <div className="bg-white rounded-lg border h-full overflow-hidden">
            <div className="relative h-full">
              <div className="absolute inset-0 overflow-auto">
                <table className="w-full table-auto min-w-[1000px]">
                  <thead className="sticky top-0 bg-gray-50 z-10 border-b">
                    <tr>
                      <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap">
                        Program
                      </th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap">
                        Process
                      </th>
                      {processData[0] &&
                        Object.keys(processData[0])
                          .filter(
                            (key) =>
                              ![
                                "timestamp",
                                "result",
                                "M_CURRENT_PROGRAM_NUMBER",
                                "Machining_Process",
                                "id",
                              ].includes(key)
                          )
                          .map((key) => (
                            <th
                              key={key}
                              className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap"
                            >
                              {formatColumnHeader(key)}
                            </th>
                          ))}
                      <th className="sticky right-0 bg-gray-50 px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-l shadow-l">
                        판정
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {processData.map((row, idx) => (
                      <tr
                        key={row.id}
                        className={`
                        ${idx % 2 === 0 ? "bg-white" : "bg-gray-50"} 
                        hover:bg-blue-50 transition-colors
                      `}
                      >
                        <td className="px-6 py-4 whitespace-nowrap">
                          {row.M_CURRENT_PROGRAM_NUMBER}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {row.Machining_Process}
                        </td>
                        {Object.entries(row)
                          .filter(
                            ([key]) =>
                              ![
                                "timestamp",
                                "result",
                                "M_CURRENT_PROGRAM_NUMBER",
                                "Machining_Process",
                                "id",
                              ].includes(key)
                          )
                          .map(([key, value]) => (
                            <td
                              key={key}
                              className="px-6 py-4 whitespace-nowrap"
                            >
                              {value}
                            </td>
                          ))}
                        <td
                          className={`
                          sticky right-0 px-6 py-4 whitespace-nowrap font-medium border-l shadow-l
                          ${
                            row.result === "합격"
                              ? "text-green-600"
                              : "text-red-600"
                          }
                          ${idx % 2 === 0 ? "bg-white" : "bg-gray-50"}
                          hover:bg-blue-50
                        `}
                        >
                          <div className="flex items-center gap-2">
                            <div
                              className={`w-3 h-3 rounded-full ${
                                row.result === "합격"
                                  ? "bg-green-500"
                                  : "bg-red-500"
                              }`}
                            />
                            {row.result}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainPage;
