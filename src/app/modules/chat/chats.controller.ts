import { RequestHandler } from 'express';
import { Chat } from './chat.model';
import { getSocketIOInstance } from '../../../socketIO';

// Helper to validate and retrieve req.user.id
const getUserId = (req: any): string => {
  if (!req.user || !req.user.id) {
    throw new Error('Unauthorized');
  }
  return req.user.id;
};

export const createChat: RequestHandler = async (req, res): Promise<void> => {
  try {
    const io = getSocketIOInstance();
    const { participants } = req.body;

    if (!participants || participants.length < 2) {
      res.status(400).json({ success: false, message: 'At least two participants are required.' });
      return;
    }

    const chat = new Chat({ participants });
    await chat.save();

    // Notify participants about the new chat
    participants.forEach((participantId: string) => {
        io.to(participantId).emit('new-chat', chat);
      });

    res.status(201).json({ success: true, chat });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
};

export const getChats: RequestHandler = async (req, res): Promise<void> => {
  try {
    const userId = getUserId(req); // Validate and get user ID

    const chats = await Chat.find({ participants: userId }).populate('participants', 'name');
    res.status(200).json({ success: true, chats });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
};
