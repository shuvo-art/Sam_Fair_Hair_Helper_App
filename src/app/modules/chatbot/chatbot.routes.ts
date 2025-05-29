import express from 'express';
import { 
  handleChatMessage,
  handlePdfUpload, 
  getAllChats, 
  getChatHistory, 
  updateChatName, 
  toggleBotResponseLikeStatus, 
  deleteChat 
} from './chatbot.controller';
import { authenticate, requireRole } from '../auth/auth.middleware';
import multer from 'multer';

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

// Admin-only PDF upload endpoint
router.post('/upload-pdf', authenticate, requireRole('admin'), upload.single('pdf'), handlePdfUpload);

// User chat message endpoint (supports text, image, and PDF)
router.post('/message', authenticate, upload.single('file'), handleChatMessage);

router.get('/all-chats', authenticate, getAllChats);
router.get('/history/:chatId', authenticate, getChatHistory);
router.put('/update-chat-name/:chatId', authenticate, updateChatName);
router.patch('/toggle-like/:chatId/:messageId', authenticate, toggleBotResponseLikeStatus);
router.delete('/delete-chat/:chatId', authenticate, deleteChat);

export default router;