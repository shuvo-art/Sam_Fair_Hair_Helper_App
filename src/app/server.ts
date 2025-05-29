import express from 'express';
import authRoutes from './modules/auth/auth.routes';
import subscriptionRoutes from './modules/subscription/subscription.routes';
import packageRoutes from './modules/subscription/package.routes';
import userRoutes from './modules/user/user.routes';
import messageRoutes from './modules/messages/messages.routes';
import chatRoutes from './modules/chat/chats.routes';
import policyRoutes from './modules/policy/policy.routes';
import dashboardRoutes from './modules/dashboard/dashboard.routes';
import chatbotRoutes from './modules/chatbot/chatbot.routes';
import problemRoutes from './modules/problem/problem.routes';
import notificationRoutes from './modules/notification/notification.routes';
import budgetRoutes from './modules/budget/budget.routes';
import { authenticate } from './modules/auth/auth.middleware';

const router = express.Router();

// Public routes
router.use('/auth', authRoutes);

// Authenticated routes
router.use('/user', authenticate, userRoutes);

router.use('/messages', authenticate, messageRoutes);
router.use('/chats', authenticate, chatRoutes);

router.use('/subscription', authenticate, subscriptionRoutes);
router.use('/package', authenticate, packageRoutes);

router.use('/notification', authenticate, notificationRoutes);
router.use('/budget', authenticate, budgetRoutes);

// Admin-only routes
router.use('/policy', authenticate, policyRoutes);
router.use('/dashboard', authenticate, dashboardRoutes);

// Chatbot routes for authenticated users
router.use('/chatbot', authenticate, chatbotRoutes);
router.use('/problem', problemRoutes);

export default router;
