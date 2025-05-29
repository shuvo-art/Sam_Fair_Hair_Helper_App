import { RequestHandler } from 'express';
import { Message } from './messages.model';
import { getSocketIOInstance } from '../../../socketIO'; // Adjust the import path as necessary

// Helper to ensure req.user is not undefined
const getUserId = (req: any): string => {
  if (!req.user || !req.user.id) {
    throw new Error('Unauthorized');
  }
  return req.user.id;
};

export const createMessage: RequestHandler = async (req, res): Promise<void> => {
  try {
    const io = getSocketIOInstance();
    const { chatId, content } = req.body;

    if (!chatId || !content) {
      res.status(400).json({ success: false, message: 'Chat ID and content are required.' });
      return;
    }

    const userId = getUserId(req);

    const message = new Message({
      chat: chatId,
      sender: userId,
      content,
    });

    await message.save();

    // Emit the new message to the chat room
    io.to(`chat-${chatId}`).emit('new-message', message);

    res.status(201).json({ success: true, message });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
};

export const getMessages: RequestHandler = async (req, res): Promise<void> => {
  try {
    const { chatId } = req.params;

    const messages = await Message.find({ chat: chatId }).populate('sender', 'name');
    res.status(200).json({ success: true, messages });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
};

export const deleteMessage: RequestHandler = async (req, res): Promise<void> => {
  try {
    const io = getSocketIOInstance();
    const { messageId } = req.params;

    const message = await Message.findByIdAndDelete(messageId);
    if (!message) {
      res.status(404).json({ success: false, message: 'Message not found.' });
      return;
    }

    // Emit the deleted message ID to the chat room
    io.to(`chat-${message.chat}`).emit('message-deleted', messageId);

    res.status(200).json({ success: true, message: 'Message deleted successfully.' });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
};

export const getMessageHistory: RequestHandler = async (req, res) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized access.' });
      return;
    }

    const messages = await Message.find({ sender: userId })
      .populate('sender', 'name')
      .sort({ createdAt: -1 });

    res.status(200).json({ success: true, messages });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
};