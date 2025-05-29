import cron from 'node-cron';
import { Subscription } from './modules/subscription/subscription.model';
import { createSubscriptionNotification } from './utils/notification.helper';
import { User } from './modules/user/user.model';
import { messaging } from './config/firebase';
import { Notification } from './modules/notification/notification.model';

// Function to check for expiries and birthdays
const checkExpiriesAndBirthdays = async () => {
  try {
    console.log('Checking for expiries and birthdays...');
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    // Trials ending soon (within 2 days)
    const trialsEndingSoon = await Subscription.find({
      trialActive: true,
      endDate: { $lte: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000) },
    });
    for (const sub of trialsEndingSoon) {
      await createSubscriptionNotification(sub.user, 'trial-ending', sub.endDate);
    }

    // Trials expired today at 10:00 PM
    const trialsExpiredToday = await Subscription.find({
      trialActive: true,
      endDate: { $gte: today, $lt: tomorrow },
    });
    for (const sub of trialsExpiredToday) {
      await createSubscriptionNotification(sub.user, 'trial-expired', sub.endDate);
    }

    // Subscriptions over today at 09:00 PM
    const subscriptionsOverToday = await Subscription.find({
      trialActive: false,
      endDate: { $gte: today, $lt: tomorrow },
    });
    for (const sub of subscriptionsOverToday) {
      await createSubscriptionNotification(sub.user, 'subscription-over', sub.endDate);
    }

    // Birthday check
    const todayMonth = today.getMonth() + 1; // getMonth() is 0-based, MongoDB $month is 1-based
    const todayDay = today.getDate();

    const usersWithBirthday = await User.aggregate([
      {
        $match: {
          birthday: { $exists: true, $ne: null },
        },
      },
      {
        $addFields: {
          birthMonth: { $month: '$birthday' },
          birthDay: { $dayOfMonth: '$birthday' },
        },
      },
      {
        $match: {
          birthMonth: todayMonth,
          birthDay: todayDay,
        },
      },
    ]);

    for (const user of usersWithBirthday) {
      const title = 'Happy Birthday!';
      const message = 'Wishing you a fantastic birthday today at 10:00 PM!';

      // Save to MongoDB
      await Notification.create({
        user: user._id,
        title,
        message,
        type: 'reminder',
        timestamp: new Date(),
      });

      // Send FCM notification
      if (user.fcmToken) {
        await messaging.send({
          notification: { title, body: message },
          token: user.fcmToken,
        });
        console.log(`Birthday notification sent to user ${user._id}`);
      }
    }
  } catch (error: any) {
    console.error('Error in expiry/birthday check:', error.message);
  }
};

// Schedule the cron job to run every day at midnight
export const startCronJobs = () => {
  cron.schedule('0 0 * * *', () => {
    console.log('Running daily expiry and birthday check...');
    checkExpiriesAndBirthdays();
  }, {
    scheduled: true,
    timezone: 'UTC',
  });

  console.log('Cron jobs scheduled.');
};