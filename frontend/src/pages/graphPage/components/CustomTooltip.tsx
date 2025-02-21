import { TooltipProps } from "recharts";
import { TrackingData } from "../types/chartTypes";

const CustomTooltip = ({ active, payload }: TooltipProps<number, string>) => {
  if (!active || !payload?.length) return null;
  const data = payload[0].payload as TrackingData;

  return (
    <div className="bg-white p-2 border border-gray-300 rounded shadow">
      <p>Time: {data.timestamp}</p>
      <p>Set: {data.setPoint.toFixed(2)}</p>
      <p>Actual: {data.actualPoint.toFixed(2)}</p>
      <p>Status: {data.passed ? "Pass" : "Fail"}</p>
      <p>Error: {(data.actualPoint - data.setPoint).toFixed(2)}</p>
    </div>
  );
};

export default CustomTooltip;
