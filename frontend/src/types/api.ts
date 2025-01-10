// src/types/api.ts
import { ProcessData } from './process';

export interface ServerResponse {
  status: string;
  data: ProcessData;
  prediction: string;
}