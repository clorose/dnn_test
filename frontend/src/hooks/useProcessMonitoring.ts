import { useState, useCallback, useEffect } from 'react';
import { ProcessData } from '../types/process';
import { ServerResponse } from '../types/api';
import ky from 'ky';

const INITIAL_URL = "https://079d-121-146-68-125.ngrok-free.app/api/process-data";  // 새 URL로 업데이트
const REMOTE_URL = "http://localhost:8080/api/process-data";

export const useProcessMonitoring = () => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [processData, setProcessData] = useState<ProcessData[]>([]);
  const [currentEndpoint, setCurrentEndpoint] = useState<string>(INITIAL_URL);

  const fetchProcessData = useCallback(async () => {
    try {
      const response = await ky.get(currentEndpoint, {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        timeout: 5000,
      }).json<ServerResponse>();

      // 응답이 성공적이면 isConnected를 true로 설정
      setIsConnected(true);

      const newData = {
        ...response.data,
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        result: (+response.prediction).toFixed(2),
      };

      setProcessData((prev) => [...prev, newData].slice(-100));
    } catch (error) {
      console.error(`Failed to fetch from ${currentEndpoint}:`, error);
      
      // 현재 엔드포인트가 실패하면 다른 엔드포인트로 전환
      const fallbackUrl = currentEndpoint === INITIAL_URL ? REMOTE_URL : INITIAL_URL;
      
      try {
        const fallbackResponse = await ky.get(fallbackUrl, {
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
          },
          timeout: 5000,
        }).json<ServerResponse>();

        setIsConnected(true);

        const newData = {
          ...fallbackResponse.data,
          id: crypto.randomUUID(),
          timestamp: new Date().toISOString(),
          result: (+fallbackResponse.prediction).toFixed(2),
        };

        setProcessData((prev) => [...prev, newData].slice(-100));
        setCurrentEndpoint(fallbackUrl);
      } catch (fallbackError) {
        console.error("Both endpoints failed:", fallbackError);
        setIsConnected(false);
      }
    }
  }, [currentEndpoint]);

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

  return { 
    isConnected, 
    setIsConnected, 
    processData,
    currentEndpoint,
  };
};