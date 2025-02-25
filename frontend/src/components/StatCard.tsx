// components/StatCard.tsx
interface StatCardProps {
  title: string;
  value: string | number;
  trend?: number;
}

const StatCard = ({ title, value, trend }: StatCardProps) => {
  return (
    <div className="bg-white rounded-lg p-4 shadow">
      <h3 className="text-gray-500 text-sm mb-2">{title}</h3>
      <div className="flex items-end justify-between">
        <div className="text-2xl font-bold">{value}</div>
        {trend !== undefined && (
          <div
            className={`text-sm ${
              trend >= 0 ? "text-green-500" : "text-red-500"
            }`}
          >
            {trend >= 0 ? "↑" : "↓"} {Math.abs(trend)}%
          </div>
        )}
      </div>
    </div>
  );
};

export default StatCard;
