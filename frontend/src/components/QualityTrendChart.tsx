// components/QualityTrendChart.tsx
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

interface QualityTrendProps {
  data: Array<{
    timestamp: string;
    prediction: number;
    passed: boolean;
  }>;
}

const QualityTrendChart = ({ data }: QualityTrendProps) => {
  return (
    <div className="bg-white rounded-lg p-4 shadow">
      <h3 className="text-lg font-semibold mb-4">품질 트렌드</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data}>
            <XAxis dataKey="timestamp" />
            <YAxis yAxisId="left" domain={[0, 1]} />
            <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Bar
              yAxisId="right"
              dataKey="passed"
              fill="#82ca9d"
              name="합격여부"
              barSize={20}
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="prediction"
              stroke="#8884d8"
              name="AI 예측"
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default QualityTrendChart;
