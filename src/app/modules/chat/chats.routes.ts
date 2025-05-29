import express from 'express';
import { createChat, getChats } from './chats.controller';
import { authenticate } from '../auth/auth.middleware';

const router = express.Router();

router.post('/', authenticate, createChat); // Create a new chat
router.get('/', authenticate, getChats); // Get all chats for a user

export default router;
