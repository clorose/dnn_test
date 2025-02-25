// components/AxisDetailsCard.tsx
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

interface AxisDetailsCardProps {
  title: string;
  current: number;
  voltage: number;
  power: number;
  data: Array<{
    time: string;
    current: number;
    voltage: number;
    power: number;
  }>;
}

const AxisDetailsCard = ({
  title,
  current,
  voltage,
  power,
  data,
}: AxisDetailsCardProps) => {
  return (
    <div className="bg-white rounded-lg p-4 shadow">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div>
          <div className="text-sm text-gray-500">Current</div>
          <div className="text-xl font-bold">{current.toFixed(2)}A</div>
        </div>
        <div>
          <div className="text-sm text-gray-500">Voltage</div>
          <div className="text-xl font-bold">{voltage.toFixed(2)}V</div>
        </div>
        <div>
          <div className="text-sm text-gray-500">Power</div>
          <div className="text-xl font-bold">{power.toFixed(2)}W</div>
        </div>
      </div>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="current"
              stroke="#8884d8"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="voltage"
              stroke="#82ca9d"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="power"
              stroke="#ffc658"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default AxisDetailsCard;
