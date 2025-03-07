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
import {
  NameType,
  ValueType,
} from "recharts/types/component/DefaultTooltipContent";

interface ScatterData {
  current: number;
  passRate: number;
}

interface AxisChartProps {
  title: string;
  data: ScatterData[];
  xLabel?: string;
  yLabel?: string;
}

interface RechartsDotProps {
  cx?: number;
  cy?: number;
  payload?: ScatterData;
}

interface RenderTooltipProps extends TooltipProps<ValueType, NameType> {
  xLabel?: string;
  yLabel?: string;
}

const CustomTooltip = ({
  active,
  payload,
  xLabel,
  yLabel,
}: TooltipProps<ValueType, NameType> & {
  xLabel?: string;
  yLabel?: string;
}) => {
  if (!active || !payload?.[0]) return null;
  const data = payload[0].payload as ScatterData;
  return (
    <div className="bg-white p-2 border border-gray-300 rounded shadow">
      <p>{`${xLabel}: ${data.current.toFixed(1)} A`}</p>
      <p>{`${yLabel}: ${data.passRate.toFixed(2)}%`}</p>
    </div>
  );
};

const RenderTooltip = ({ xLabel, yLabel, ...props }: RenderTooltipProps) => (
  <CustomTooltip {...props} xLabel={xLabel} yLabel={yLabel} />
);

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

const CustomScatter = ({ title, data, xLabel, yLabel }: AxisChartProps) => {
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
              domain={["dataMin", "dataMax"]}
              label={{ value: xLabel, position: "bottom" }}
            />
            <YAxis
              type="number"
              dataKey="passRate"
              domain={[0, 100]}
              label={{ value: yLabel, angle: -90, position: "left" }}
            />
            <Tooltip
              content={<RenderTooltip xLabel={xLabel} yLabel={yLabel} />}
            />
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
