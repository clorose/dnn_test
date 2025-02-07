// path: D:\DNN_test\nodejs\app.js
const express = require('express');
const cors = require('cors');
const pool = require('./db');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  const name = process.env.NAME || 'world';
  res.send("Hello, "+name);
  console.log(name);
  
});

app.get('/test', async (req, res) => {
  try {
    console.log('DB 연결 시도...');
    console.log('DB 설정:', {
      host: process.env.DB_HOST,
      port: process.env.DB_PORT,
      database: process.env.DB_NAME,
      user: process.env.DB_USER
    });
    
    const result = await pool.query('SELECT NOW()');
    console.log('쿼리 결과:', result.rows[0]);
    res.json(result.rows[0]);
  } catch (err) {
    console.error('DB 에러:', err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/machine', async(req, res) => {
  try{
    const machine = await pool.query('SELECT * FROM machine_process');
    console.log("데이터 조회", machine.rows[0]);
    res.json(machine.rows[0]);
  } catch(err){
    console.error('DB 에러:', err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log('환경변수 확인:', {
    host: process.env.DB_HOST,
    port: process.env.DB_PORT,
    database: process.env.DB_NAME,
    user: process.env.DB_USER
  });
});