import { Notification } from '../modules/notification/notification.model';
import { Types } from 'mongoose';
import { messaging } from '../config/firebase'; // Import Firebase messaging

export const createSubscriptionNotification = async (
  userId: Types.ObjectId,
  type: 'trial-ending' | 'subscription-canceled' | 'trial-expired' | 'subscription-over',
  date?: Date
): Promise<void> => {
  try {
    let title = '';
    let message = '';

    switch (type) {
      case 'trial-ending':
        title = 'Your trial is ending soon';
        message = `Your free trial will end on ${date?.toDateString()}. Upgrade now to avoid interruption.`;
        break;
      case 'subscription-canceled':
        title = 'Your subscription was canceled';
        message = 'You no longer have an active subscription. Please renew to regain access.';
        break;
      case 'trial-expired':
        title = 'Your Trial is expired';
        message = `Your trial expired on ${date?.toDateString()} at 10:00 PM.`;
        break;
      case 'subscription-over':
        title = 'Your Subscription is over';
        message = `Your subscription ended on ${date?.toDateString()} at 09:00 PM.`;
        break;
    }

    // Save notification to MongoDB
    const notification = await Notification.create({
      user: userId,
      title,
      message,
      type: 'subscription',
      timestamp: new Date(),
    });

    // Send push notification via Firebase
    const userFcmToken = await getUserFcmToken(userId); // You'll need to implement this
    if (userFcmToken) {
      const fcmMessage = {
        notification: {
          title,
          body: message,
        },
        token: userFcmToken,
      };

      await messaging.send(fcmMessage);
      console.log(`Push notification sent to user ${userId}`);
    } else {
      console.log(`No FCM token found for user ${userId}`);
    }
  } catch (error) {
    console.error('Failed to create/send notification:', error);
  }
};

// Placeholder function to get user's FCM token (implement this based on your client-side setup)
async function getUserFcmToken(userId: Types.ObjectId): Promise<string | null> {
  // Assume FCM tokens are stored in the User model or a separate collection
  // For now, return a dummy token or fetch from your database
  const user = await import('../modules/user/user.model').then(m => m.User.findById(userId));
  return user?.fcmToken || null; // Add fcmToken field to User model
}