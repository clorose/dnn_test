export const useResultStatus = (prediction: string) => {
  const value = Number(prediction);

  if (value >= 0.8) {
    return {
      label: "합격 상",
      dotColor: "bg-green-500", // 진한 초록색
      textColor: "text-green-500",
    };
  } else if (value >= 0.65) {
    return {
      label: "합격 중",
      dotColor: "bg-lime-500", // 라임색
      textColor: "text-lime-500",
    };
  } else if (value >= 0.5) {
    return {
      label: "합격 하",
      dotColor: "bg-yellow-500", // 노란색
      textColor: "text-yellow-500",
    };
  } else if (value >= 0.35) {
    return {
      label: "불합격 하",
      dotColor: "bg-amber-500", // 황색
      textColor: "text-amber-500",
    };
  } else if (value >= 0.2) {
    return {
      label: "불합격 중",
      dotColor: "bg-orange-500", // 주황색
      textColor: "text-orange-500",
    };
  } else {
    return {
      label: "불합격 상",
      dotColor: "bg-red-500", // 빨간색
      textColor: "text-red-500",
    };
  }
};