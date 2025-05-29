import express from 'express';
import { getAllNotifications, markNotificationAsRead } from './notification.controller';
import { authenticate } from '../auth/auth.middleware';

const router = express.Router();

// Get all notifications for the authenticated user
router.get('/', authenticate, getAllNotifications);
// Mark a specific notification as read
router.put('/:id/read', authenticate, markNotificationAsRead);


export default router;
