import { Server as HttpServer } from 'http';
import { Server } from 'socket.io';
import { getUserFromToken } from './app/helpers/auth.helper';
import { Message } from './app/modules/messages/messages.model';
import { Chat } from './app/modules/chat/chat.model';
import { IUser } from './app/modules/user/user.model';

let io: Server | null = null;

export const initializeSocketIO = (server: HttpServer): Server => {
  io = new Server(server, {
    cors: { origin: '*' },
  });

  const onlineUsers = new Map<string, string>();

  io.on('connection', async (socket) => {
    console.log(`User connected: ${socket.id}`);

    const token = socket.handshake.auth.token;
    const user = await getUserFromToken(token) as IUser;

    if (!user) {
      socket.emit('auth-error', { message: 'Invalid token' });
      return socket.disconnect();
    }

    // Ensure _id is string
    const userId = (user._id as string).toString();
    onlineUsers.set(userId, socket.id);
    io?.emit('online-users', Array.from(onlineUsers.keys()));

    socket.join(userId);

    // Event handlers for chat and messages
    socket.on('join-room', (chatId) => {
      socket.join(`chat-${chatId}`);
      console.log(`User ${userId} joined chat room: chat-${chatId}`);
    });

    socket.on('send-message', async ({ chatId, content }) => {
      if (!chatId || !content) {
        socket.emit('error', { message: 'Chat ID and message content are required.' });
        return;
      }

      const message = new Message({ chat: chatId, sender: userId, content });
      await message.save();
      io?.to(`chat-${chatId}`).emit('receive-message', message);
    });

    socket.on('admin-join-user', (userId) => {
      socket.join(userId);
      console.log(`Admin joined user room: ${userId}`);
    });
    
    socket.on('request-user-data', async (userId) => {
      const messages = await Message.find({ sender: userId }).populate('chat');
      const chats = await Chat.find({ participants: userId }).populate('participants', 'name');
      socket.emit('user-data', { messages, chats });
    });

    socket.on('disconnect', () => {
      onlineUsers.delete(userId);
      io?.emit('online-users', Array.from(onlineUsers.keys()));
      console.log(`User disconnected: ${socket.id}`);
    });
  });

  return io;
};

export const getSocketIOInstance = (): Server => {
  if (!io) throw new Error('Socket.IO instance not initialized');
  return io;
};