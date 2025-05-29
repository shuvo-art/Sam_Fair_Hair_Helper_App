import express from 'express';
import { reportProblem } from './problem.controller';

const router = express.Router();

// Endpoint for reporting a problem
router.post('/report', reportProblem);

export default router;
