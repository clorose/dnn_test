// path: frontend/src/pages/graphPage/components/CustomScatter.tsx
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  TooltipProps,
} from "recharts";

interface ScatterData {
  current: number;
  passRate: number;
}

interface AxisChartProps {
  title: string;
  data: ScatterData[];
}

interface RechartsDotProps {
  cx?: number;
  cy?: number;
  payload?: ScatterData;
}

const CustomTooltip = ({ active, payload }: TooltipProps<number, string>) => {
  if (!active || !payload?.length) return null;
  const data = payload[0].payload as ScatterData;
  return (
    <div className="bg-white p-2 border border-gray-300 rounded shadow">
      <p>{`Output Current: ${data.current.toFixed(1)} A`}</p>
      <p>{`합격률: ${data.passRate.toFixed(2)}%`}</p>
    </div>
  );
};

const CustomShape = ({ cx, cy, payload }: RechartsDotProps): JSX.Element => {
  if (typeof cx === "undefined" || typeof cy === "undefined" || !payload) {
    return <circle cx={0} cy={0} r={0} fill="none" />;
  }
  return (
    <circle
      cx={cx}
      cy={cy}
      r={4}
      fill={payload.passRate >= 80 ? "#4CAF50" : "#f44336"}
    />
  );
};

const CustomScatter = ({ title, data }: AxisChartProps) => {
  return (
    <>
      <h2 className="text-xl font-bold">{title}</h2>
      <div className="h-[300px] mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 30, left: 60, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="current"
              label={{ value: "Output Current (A)", position: "bottom" }}
            />
            <YAxis
              type="number"
              dataKey="passRate"
              domain={[0, 100]}
              label={{ value: "합격률 (%)", angle: -90, position: "left" }}
            />
            <Tooltip content={CustomTooltip} />
            <Scatter
              data={data}
              shape={CustomShape}
              isAnimationActive={false}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </>
  );
};

export default CustomScatter;
