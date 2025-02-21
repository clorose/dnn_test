// path: frontend/src/pages/graphPage/components/TrackingChart.tsx
import {
  TrackingData,
  TrackingChartProps,
  FailRange,
} from "../types/chartTypes";
import CustomTooltip from "./CustomTooltip";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceArea,
} from "recharts";

const TrackingChart = ({ title, data, yAxisLabel }: TrackingChartProps) => {
  const getFailRanges = (data: TrackingData[]): FailRange[] => {
    const ranges: FailRange[] = [];
    let start: number | null = null;

    data.forEach((point, index) => {
      if (!point.passed && start === null) {
        start = index;
      } else if (
        (point.passed || index === data.length - 1) &&
        start !== null
      ) {
        ranges.push({ start, end: index });
        start = null;
      }
    });

    return ranges;
  };

  const generateHash = (range: FailRange) => {
    return btoa(`${range.start}-${range.end}`).replace(/=/g, "");
  };

  const failRanges = getFailRanges(data);

  return (
    <div className="bg-white shadow-lg rounded-lg p-4">
      <h2 className="text-xl font-bold mb-4">{title}</h2>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 30, left: 60, bottom: 25 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp"
              label={{ value: "Time", position: "bottom" }}
            />
            <YAxis
              label={{ value: yAxisLabel, angle: -90, position: "left" }}
            />
            {/* 불합격 구간 표시 */}
            {failRanges.map((range) => (
              <ReferenceArea
                key={generateHash(range)}
                x1={data[range.start].timestamp}
                x2={data[range.end].timestamp}
                fill="#ff000020"
              />
            ))}
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey="setPoint"
              stroke="#2196F3"
              name="Set Point"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="actualPoint"
              stroke="#4CAF50"
              name="Actual Point"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TrackingChart;
