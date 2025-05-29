import express from 'express';
import { 
  createMessage, 
  getMessages, 
  deleteMessage, 
  getMessageHistory 
} from './messages.controller';
import { authenticate } from '../auth/auth.middleware';

const router = express.Router();

router.post('/', authenticate, createMessage);

router.get('/chat/:chatId', authenticate, getMessages);

router.delete('/:messageId', authenticate, deleteMessage);

router.get('/history', authenticate, getMessageHistory);

export default router;
