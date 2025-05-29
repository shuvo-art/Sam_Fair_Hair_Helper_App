import express from 'express';
import { getPolicy, updatePolicy, createPolicy } from './policy.controller';
import { authenticate, requireRole } from '../auth/auth.middleware';

const router = express.Router();

// Admin-only access to policies
router.get('/:type', authenticate, getPolicy);
router.put('/:type', authenticate, requireRole('admin'), updatePolicy);
router.post('/', authenticate, requireRole('admin'), createPolicy);

export default router;
