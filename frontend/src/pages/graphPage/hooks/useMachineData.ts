// path: frontend/src/pages/graphPage/hooks/useMachineData.ts
import { useState, useEffect } from 'react';
import { MachineData, ScatterData } from '../types/machineData';
import { TrackingData } from '../types/chartTypes';
import ky from 'ky';

export const useMachineData = () => {
  const [data, setData] = useState<MachineData | null>(null);
  const [qualityCounts, setQualityCounts] = useState({ pass: 0, fail: 0 });
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [outputCurrentData, setOutputCurrentData] = useState<{
    X: ScatterData[];
    Y: ScatterData[];
  }>({ X: [], Y: [] });

  const [trackingData, setTrackingData] = useState<{
    XPosition: TrackingData[];
    YPosition: TrackingData[];
    XVelocity: TrackingData[];
    YVelocity: TrackingData[];
  }>({
    XPosition: [],
    YPosition: [],
    XVelocity: [],
    YVelocity: [],
  });

  const [powerCurrentData, setPowerCurrentData] = useState<{
    X: ScatterData[];
    Y: ScatterData[];
  }>({ X: [], Y: [] });

  const updateOutputCurrentData = (newData: MachineData) => {
    const currentPoint = {
      X: {
        current: newData.X.OutputCurrent,
        passRate: newData.quality.ai_prediction !== null
          ? newData.quality.ai_prediction * 100
          : 0,
      },
      Y: {
        current: newData.Y.OutputCurrent,
        passRate: newData.quality.ai_prediction !== null
          ? newData.quality.ai_prediction * 100
          : 0,
      },
    };

    setOutputCurrentData((prev) => {
      if (prev.X.length >= 10000) {
        return {
          X: [...prev.X.slice(1), currentPoint.X],
          Y: [...prev.Y.slice(1), currentPoint.Y],
        };
      }
      return {
        X: [...prev.X, currentPoint.X],
        Y: [...prev.Y, currentPoint.Y],
      };
    });
  };

  const updatePowerCurrentData = (newData: MachineData) => {
    const currentPoint = {
      X: {
        current: newData.X.OutputPower,
        passRate: newData.quality.ai_prediction !== null
          ? newData.quality.ai_prediction * 100
          : 0,
      },
      Y: {
        current: newData.Y.OutputPower,
        passRate: newData.quality.ai_prediction !== null
          ? newData.quality.ai_prediction * 100
          : 0,
      }
    };

    setPowerCurrentData((prev) => {
      if (prev.X.length >= 10000) {
        return {
          X: [...prev.X.slice(1), currentPoint.X],
          Y: [...prev.Y.slice(1), currentPoint.Y],
        };
      }
      return {
        X: [...prev.X, currentPoint.X],
        Y: [...prev.Y, currentPoint.Y],
      };
    });
  };

  const updateQualityCounts = (newData: MachineData) => {
    if (newData.quality.passed !== null) {
      setQualityCounts((prev) => ({
        pass: prev.pass + (newData.quality.passed ? 1 : 0),
        fail: prev.fail + (newData.quality.passed ? 0 : 1),
      }));
    }
  };

  const updateTrackingData = (newData: MachineData) => {
    const currentTime = new Date().toLocaleTimeString();
    const trackingPoint = {
      XPosition: {
        timestamp: currentTime,
        setPoint: newData.X.SetPosition,
        actualPoint: newData.X.ActualPosition,
        passed: newData.quality.passed || false,
      },
      YPosition: {
        timestamp: currentTime,
        setPoint: newData.Y.SetPosition,
        actualPoint: newData.Y.ActualPosition,
        passed: newData.quality.passed || false,
      },
      XVelocity: {
        timestamp: currentTime,
        setPoint: newData.X.SetVelocity,
        actualPoint: newData.X.ActualVelocity,
        passed: newData.quality.passed || false,
      },
      YVelocity: {
        timestamp: currentTime,
        setPoint: newData.Y.SetVelocity,
        actualPoint: newData.Y.ActualVelocity,
        passed: newData.quality.passed || false,
      },
    };

    setTrackingData((prev) => {
      const maxLength = 100;
      return {
        XPosition: [
          ...(prev.XPosition.length >= maxLength ? prev.XPosition.slice(1) : prev.XPosition),
          trackingPoint.XPosition,
        ],
        YPosition: [
          ...(prev.YPosition.length >= maxLength ? prev.YPosition.slice(1) : prev.YPosition),
          trackingPoint.YPosition,
        ],
        XVelocity: [
          ...(prev.XVelocity.length >= maxLength ? prev.XVelocity.slice(1) : prev.XVelocity),
          trackingPoint.XVelocity,
        ],
        YVelocity: [
          ...(prev.YVelocity.length >= maxLength ? prev.YVelocity.slice(1) : prev.YVelocity),
          trackingPoint.YVelocity,
        ],
      };
    });
  };

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await ky("http://localhost:3000/test");
        setIsConnected(response.ok);
      } catch (error) {
        setIsConnected(false);
        setError(`서버 연결 실패: ${error instanceof Error ? error.message : "알 수 없는 오류"}`);
      }
    };
    checkConnection();
  }, []);

  useEffect(() => {
    if (!isConnected) return;

    const fetchData = async () => {
      try {
        const response = await ky("http://localhost:3000/machine");
        if (!response.ok) throw new Error("데이터 조회 실패");

        const newData: MachineData = await response.json();
        console.log(`newData: ${JSON.stringify(newData)}`);
        setData(newData);
        updateOutputCurrentData(newData);
        updateQualityCounts(newData);
        updateTrackingData(newData);
        updatePowerCurrentData(newData);
      } catch (error) {
        setError(error instanceof Error ? error.message : "데이터 조회 중 오류 발생");
      }
    };

    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [isConnected]);

  return {
    data,
    qualityCounts,
    error,
    isConnected,
    outputCurrentData,
    trackingData,
    powerCurrentData
  };
};