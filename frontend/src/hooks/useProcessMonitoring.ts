import { useState, useCallback, useEffect } from 'react';
import { ProcessData } from '../types/process';
import { ServerResponse } from '../types/api';
import ky from 'ky';

const LOCAL_URL = "http://localhost:8080/api/process-data";
const REMOTE_URL = "http://your-ip-address/api/process-data";

export const useProcessMonitoring = () => {
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [processData, setProcessData] = useState<ProcessData[]>([]);
  const [currentEndpoint, setCurrentEndpoint] = useState<string>(LOCAL_URL);

  const fetchProcessData = useCallback(async () => {
    try {
      // 현재 설정된 엔드포인트로 시도
      const response = await ky
        .get(currentEndpoint)
        .json<ServerResponse>();

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
      const fallbackUrl = currentEndpoint === LOCAL_URL ? REMOTE_URL : LOCAL_URL;
      
      try {
        const fallbackResponse = await ky
          .get(fallbackUrl)
          .json<ServerResponse>();

        const newData = {
          ...fallbackResponse.data,
          id: crypto.randomUUID(),
          timestamp: new Date().toISOString(),
          result: (+fallbackResponse.prediction).toFixed(2),
        };

        setProcessData((prev) => [...prev, newData].slice(-100));
        setCurrentEndpoint(fallbackUrl); // 성공한 엔드포인트로 변경
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
    currentEndpoint,  // 현재 사용 중인 엔드포인트 정보도 반환
  };
};