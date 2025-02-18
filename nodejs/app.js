// path: ~/Develop/dnn_test/nodejs/app.js
const express = require('express');
const cors = require('cors');
const pool = require('./db');
const axios = require('axios');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

// FastAPI 서버 URL 설정
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

// 루트 엔드포인트
app.get('/', (req, res) => {
  const name = process.env.NAME || 'world';
  res.send("Hello, " + name);
});

// DB 연결 테스트 엔드포인트
app.get('/test', async (req, res) => {
  try {
    const result = await pool.query('SELECT NOW()');
    res.json(result.rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 기계 데이터 엔드포인트
app.get('/machine', async (req, res) => {
  try {
    const query = `
      SELECT *
      FROM machine
      ORDER BY RANDOM() 
      LIMIT 1
    `;
    const result = await pool.query(query);
    const rawData = result.rows[0];
    console.log(rawData);

    if (!rawData) {
      return res.status(404).json({ error: "데이터가 없습니다" });
    }

    const mappedProcess = mapMachiningProcess(rawData.Machining_Process);

    const formattedData = {
      timestamp: rawData.timestamp,
      X: {
        ActualPosition: rawData.X1_ActualPosition,
        ActualVelocity: rawData.X1_ActualVelocity,
        ActualAcceleration: rawData.X1_ActualAcceleration,
        SetPosition: rawData.X1_CommandPosition,
        SetVelocity: rawData.X1_CommandVelocity,
        SetAcceleration: rawData.X1_CommandAcceleration,
        CurrentFeedback: rawData.X1_CurrentFeedback,
        DCBusVoltage: rawData.X1_DCBusVoltage,
        OutputCurrent: rawData.X1_OutputCurrent,
        OutputVoltage: rawData.X1_OutputVoltage,
        OutputPower: rawData.X1_OutputPower
      },
      Y: {
        ActualPosition: rawData.Y1_ActualPosition,
        ActualVelocity: rawData.Y1_ActualVelocity,
        ActualAcceleration: rawData.Y1_ActualAcceleration,
        SetPosition: rawData.Y1_CommandPosition,
        SetVelocity: rawData.Y1_CommandVelocity,
        SetAcceleration: rawData.Y1_CommandAcceleration,
        CurrentFeedback: rawData.Y1_CurrentFeedback,
        DCBusVoltage: rawData.Y1_DCBusVoltage,
        OutputCurrent: rawData.Y1_OutputCurrent,
        OutputVoltage: rawData.Y1_OutputVoltage,
        OutputPower: rawData.Y1_OutputPower
      },
      Z: {
        ActualPosition: rawData.Z1_ActualPosition,
        ActualVelocity: rawData.Z1_ActualVelocity,
        ActualAcceleration: rawData.Z1_ActualAcceleration,
        SetPosition: rawData.Z1_CommandPosition,
        SetVelocity: rawData.Z1_CommandVelocity,
        SetAcceleration: rawData.Z1_CommandAcceleration,
        CurrentFeedback: rawData.Z1_CurrentFeedback,
        DCBusVoltage: rawData.Z1_DCBusVoltage,
        OutputCurrent: rawData.Z1_OutputCurrent,
        OutputVoltage: rawData.Z1_OutputVoltage
      },
      S: {
        ActualPosition: rawData.S1_ActualPosition,
        ActualVelocity: rawData.S1_ActualVelocity,
        ActualAcceleration: rawData.S1_ActualAcceleration,
        SetPosition: rawData.S1_CommandPosition,
        SetVelocity: rawData.S1_CommandVelocity,
        SetAcceleration: rawData.S1_CommandAcceleration,
        CurrentFeedback: rawData.S1_CurrentFeedback,
        DCBusVoltage: rawData.S1_DCBusVoltage,
        OutputCurrent: rawData.S1_OutputCurrent,
        OutputVoltage: rawData.S1_OutputVoltage,
        OutputPower: rawData.S1_OutputPower,
        SystemInertia: rawData.S1_SystemInertia
      },
      M: {
        CURRENT_PROGRAM_NUMBER: rawData.M1_CURRENT_PROGRAM_NUMBER,
        sequence_number: rawData.M1_sequence_number,
        CURRENT_FEEDRATE: rawData.M1_CURRENT_FEEDRATE
      },
      status: rawData.Machining_Process,
      quality: {
        ai_prediction: null,
        passed: null
      },
      mapped_process: mappedProcess
    };

    const features = [
      rawData.X1_ActualPosition, rawData.X1_ActualVelocity, rawData.X1_ActualAcceleration,
      rawData.X1_CommandPosition, rawData.X1_CommandVelocity, rawData.X1_CommandAcceleration,
      rawData.X1_CurrentFeedback, rawData.X1_DCBusVoltage, rawData.X1_OutputCurrent,
      rawData.X1_OutputVoltage, rawData.X1_OutputPower,
      rawData.Y1_ActualPosition, rawData.Y1_ActualVelocity, rawData.Y1_ActualAcceleration,
      rawData.Y1_CommandPosition, rawData.Y1_CommandVelocity, rawData.Y1_CommandAcceleration,
      rawData.Y1_CurrentFeedback, rawData.Y1_DCBusVoltage, rawData.Y1_OutputCurrent,
      rawData.Y1_OutputVoltage, rawData.Y1_OutputPower,
      rawData.Z1_ActualPosition, rawData.Z1_ActualVelocity, rawData.Z1_ActualAcceleration,
      rawData.Z1_CommandPosition, rawData.Z1_CommandVelocity, rawData.Z1_CommandAcceleration,
      rawData.Z1_CurrentFeedback, rawData.Z1_DCBusVoltage, rawData.Z1_OutputCurrent,
      rawData.Z1_OutputVoltage,
      rawData.S1_ActualPosition, rawData.S1_ActualVelocity, rawData.S1_ActualAcceleration,
      rawData.S1_CommandPosition, rawData.S1_CommandVelocity, rawData.S1_CommandAcceleration,
      rawData.S1_CurrentFeedback, rawData.S1_DCBusVoltage, rawData.S1_OutputCurrent,
      rawData.S1_OutputVoltage, rawData.S1_OutputPower, rawData.S1_SystemInertia,
      parseFloat(rawData.M1_CURRENT_PROGRAM_NUMBER), parseFloat(rawData.M1_sequence_number),
      parseFloat(rawData.M1_CURRENT_FEEDRATE), mappedProcess
    ];

    const cleanFeatures = features.map(value => (value === undefined || value === null ? 0 : value));

    try {
      const aiResponse = await axios.post(`${FASTAPI_URL}/predict`, { features: cleanFeatures });
      formattedData.quality.ai_prediction = aiResponse.data.prediction[0][0];
      formattedData.quality.passed = aiResponse.data.prediction[0][0] >= 0.8;
    } catch (aiError) {
      formattedData.quality.ai_prediction = 0;
      formattedData.quality.passed = false;
    }

    res.json(formattedData);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

function mapMachiningProcess(process) {
  const machiningProcessMap = {
    "Prep": 0.0, "Layer 1 Up": 1.0, "Layer 1 Down": 2.0,
    "Layer 2 Up": 3.0, "Layer 2 Down": 4.0, "Layer 3 Up": 5.0,
    "Layer 3 Down": 6.0, "Repositioning": 7.0, "End": 8.0,
    "Starting": 9.0
  };
  return machiningProcessMap[process] ?? 0;
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
