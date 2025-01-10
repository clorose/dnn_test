// path: frontend/src/pages/mainPage/mainPage.tsx
import { useRef, useState, useCallback, useEffect } from "react";
import { useProcessMonitoring } from "../../hooks/useProcessMonitoring";

const MainPage = () => {
  const { isConnected, setIsConnected, processData } = useProcessMonitoring();
  const tableRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  const handleScroll = useCallback(() => {
    if (!tableRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = tableRef.current;
    if (scrollHeight - scrollTop - clientHeight > 50) {
      setAutoScroll(false);
    } else {
      setAutoScroll(true);
    }
  }, []);

  useEffect(() => {
    if (autoScroll && tableRef.current) {
      tableRef.current.scrollTop = tableRef.current.scrollHeight;
    }
  }, [processData, autoScroll]);

  const formatColumnHeader = (col: string): string => {
    return col
      .split("_")
      .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
      .join(" ");
  };

  return (
    <div className="flex items-center justify-center w-screen h-screen bg-gray-100">
      <div className="w-[90vw] h-[95vh] bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="h-16 px-6 bg-white border-b flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-800">
            CNC 공정 불량진단 데이터 모니터링
          </h1>
          <button
            className={`px-6 py-2 rounded-lg font-medium transition-colors shadow-sm
              ${
                isConnected
                  ? "bg-green-500 hover:bg-green-600 text-white"
                  : "bg-red-500 hover:bg-red-600 text-white"
              }`}
            onClick={() => setIsConnected(!isConnected)}
          >
            {isConnected ? "가동중" : "가동 중지"}
          </button>
        </div>

        <div className="p-6 h-[calc(100%-4rem)]">
          <div className="bg-white rounded-lg border h-full overflow-hidden">
            <div className="relative h-full">
              <div
                ref={tableRef}
                className="absolute inset-0 overflow-auto"
                onScroll={handleScroll}
              >
                <table className="w-full table-auto min-w-[1000px]">
                  <thead className="sticky top-0 bg-gray-50 z-10 border-b">
                    <tr>
                      {processData[0] &&
                        Object.keys(processData[0])
                          .filter(
                            (key) =>
                              !["timestamp", "result", "id"].includes(key)
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
                        AI 판정
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
                        {Object.entries(row)
                          .filter(
                            ([key]) =>
                              !["timestamp", "result", "id"].includes(key)
                          )
                          .map(([key, value]) => (
                            <td
                              key={key}
                              className="px-6 py-4 whitespace-nowrap"
                            >
                              {typeof value === "number" &&
                              !Number.isInteger(value)
                                ? value.toFixed(4)
                                : value}
                            </td>
                          ))}
                        <td
                          className={`
                            sticky right-0 px-6 py-4 whitespace-nowrap font-medium border-l shadow-l
                            ${
                              // row.result && parseFloat(row.result) >= 0.5
                              row.result?.startsWith("합격")
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
                                row.result?.startsWith("합격")
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
