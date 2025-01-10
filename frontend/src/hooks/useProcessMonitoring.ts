// path: frontend/src/hooks/useProcessMonitoring.ts
import { useState, useCallback, useEffect } from 'react';
import { ProcessData } from '../types/process';
import { ServerResponse } from '../types/api';
import ky from 'ky';

export const useProcessMonitoring = () => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [processData, setProcessData] = useState<ProcessData[]>([]);

  const fetchProcessData = useCallback(async () => {
    try {
      const response = await ky
        .get("http://localhost:8080/api/process-data")
        .json<ServerResponse>();

      const newData = {
        ...response.data,
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        // result: (+response.prediction).toFixed(2),
        result: `${response.prediction >= "0.5" ? "합격" : "불합격"}(${(+response.prediction).toFixed(2)})`,
      };

      setProcessData((prev) => [...prev, newData].slice(-100));
    } catch (error) {
      console.error("Failed to fetch process data:", error);
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout | undefined;

    if (isConnected) {
      fetchProcessData();
      interval = setInterval(fetchProcessData, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isConnected, fetchProcessData]);

  return { isConnected, setIsConnected, processData };
};