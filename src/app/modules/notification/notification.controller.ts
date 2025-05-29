import { Request, Response } from 'express';
import { Notification } from './notification.model';

export const getAllNotifications = async (req: Request, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;

    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    const notifications = await Notification.find({ user: userId }).sort({ timestamp: -1 });

    res.status(200).json({
      success: true,
      notifications: notifications.map(n => ({
        id: n._id,
        title: n.title,
        message: n.message,
        type: n.type,
        read: n.read,
        timestamp: n.timestamp,
      }))
    });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
};


export const markNotificationAsRead = async (req: Request, res: Response): Promise<void> => {
    try {
      const { id } = req.params;
      const userId = req.user?.id;
  
      const notification = await Notification.findOneAndUpdate(
        { _id: id, user: userId },
        { read: true },
        { new: true }
      );
  
      if (!notification) {
        res.status(404).json({ success: false, message: 'Notification not found' });
        return;
      }
  
      res.status(200).json({ success: true, message: 'Notification marked as read' });
    } catch (error: any) {
      res.status(500).json({ success: false, message: error.message });
    }
  };