// 가로 막대 그래프를 위한 바 컴포넌트
interface BarProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  payload?: {
    name: string;
    count: number;
  };
}

const CustomBar = (props: BarProps) => {
  const { x = 0, y = 0, width = 0, height = 0, payload } = props;

  // 가로 막대이므로 width가 길이를 결정
  return (
    <rect
      x={x}
      y={y}
      width={width}
      height={height}
      fill={payload?.name === "Pass" ? "#4CAF50" : "#f44336"}
    />
  );
};

export default CustomBar;
