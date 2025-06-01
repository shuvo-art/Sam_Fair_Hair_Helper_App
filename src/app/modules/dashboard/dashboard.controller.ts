import { Request, Response } from 'express';
import mongoose from 'mongoose';
import { User, IUser } from '../user/user.model';
import { ChatHistory } from '../chatbot/chatHistory.model';
import { Subscription } from '../subscription/subscription.model';
import { Package } from '../subscription/package.model';
import { uploadImage, uploadHeroImage, getHeroImages, deleteHeroImage } from '../../utils/s3';
//import { uploadImage, uploadHeroImage, getHeroImages, deleteHeroImage } from '../../utils/cloudinary';
import fs from 'fs/promises';
import Stripe from 'stripe';
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY as string, {
  apiVersion: '2024-12-18.acacia',
});

// Task 1: Get all users with subscription details
export const getAllUsers = async (req: Request, res: Response): Promise<void> => {
  try {
    // Select profileImage along with other fields
    const users = await User.find().select('name email role createdAt profileImage');
    const subscriptions = await Subscription.find().populate('package');

    const userData = users.map((user) => {
      const typedUser = user as IUser & { _id: mongoose.Types.ObjectId };
      const subscription = subscriptions.find((sub) => sub.user.toString() === typedUser._id.toString());
      const packageDetails = subscription?.package as any;
      const amount = subscription?.amountPaid || (packageDetails ? packageDetails.amount / 100 : 0);
      return {
        _id: typedUser._id.toString(),
        name: typedUser.name,
        email: typedUser.email,
        subscriptionType: packageDetails ? packageDetails.subscriptionType : 'None',
        status: subscription
          ? subscription.trialActive
            ? 'Trial'
            : subscription.endDate > new Date()
            ? 'Active'
            : 'Expired'
          : 'None',
        amount: `$${amount.toFixed(2)}`,
        startDate: subscription ? subscription.startDate.toISOString() : null, // Add startDate
        endDate: subscription ? subscription.endDate.toISOString() : null, // Add endDate
        profileImage: typedUser.profileImage || null, // Include profileImage (null if not set)
      };
    });

    // Task 2: Calculate total revenue, total users, active users
    const totalUsers = users.length;
    const totalActiveUsers = subscriptions.filter(
      (sub) => sub.endDate > new Date() && !sub.trialActive
    ).length;
    const totalRevenue = subscriptions.reduce((sum, sub) => {
      const pkg = sub.package as any;
      const amount = sub.amountPaid || (pkg.amount / 100);
      return sub.endDate > new Date() && !sub.trialActive ? sum + amount : sum;
    }, 0);

    // Task 2: Monthly revenue and user growth (filterable by year)
    const year = req.query.year ? parseInt(req.query.year as string) : new Date().getFullYear();

    // Monthly revenue aggregation
    const monthlyRevenue = await Subscription.aggregate([
      { $match: { startDate: { $gte: new Date(`${year}-01-01`), $lte: new Date(`${year}-12-31`) } } },
      {
        $lookup: {
          from: 'packages',
          localField: 'package',
          foreignField: '_id',
          as: 'package',
        },
      },
      { $unwind: '$package' },
      {
        $group: {
          _id: { $month: '$startDate' },
          revenue: {
            $sum: {
              $cond: [
                { $and: [{ $gt: ['$endDate', new Date()] }, { $eq: ['$trialActive', false] }] },
                { $ifNull: ['$amountPaid', { $divide: ['$package.amount', 100] }] },
                0
              ]
            }
          },
        },
      },
      { $sort: { '_id': 1 } },
    ]);

    // Monthly user growth aggregation
    const monthlyUserGrowth = await User.aggregate([
      { $match: { createdAt: { $gte: new Date(`${year}-01-01`), $lte: new Date(`${year}-12-31`) } } },
      {
        $group: {
          _id: { $month: '$createdAt' },
          count: { $sum: 1 },
        },
      },
      { $sort: { '_id': 1 } },
    ]);

    // Transform revenue data to desired format
    const monthNames = [
      "January", "February", "March", "April", "May", "June",
      "July", "August", "September", "October", "November", "December"
    ];
    const revenueChart: { [key: string]: number }[] = [{
      January: 0, February: 0, March: 0, April: 0, May: 0, June: 0,
      July: 0, August: 0, September: 0, October: 0, November: 0, December: 0
    }];
    monthlyRevenue.forEach((item) => {
      const monthIndex = item._id - 1; // Convert 1-based month to 0-based index
      if (monthIndex >= 0 && monthIndex < 12) {
        revenueChart[0][monthNames[monthIndex]] = item.revenue || 0;
      }
    });

    // Transform growth data to desired format
    const growth: { [key: string]: number }[] = [{
      January: 0, February: 0, March: 0, April: 0, May: 0, June: 0,
      July: 0, August: 0, September: 0, October: 0, November: 0, December: 0
    }];
    monthlyUserGrowth.forEach((item) => {
      const monthIndex = item._id - 1; // Convert 1-based month to 0-based index
      if (monthIndex >= 0 && monthIndex < 12) {
        growth[0][monthNames[monthIndex]] = item.count || 0;
      }
    });

    res.status(200).json({
      success: true,
      users: userData,
      totalRevenue: totalRevenue.toFixed(2),
      totalUsers,
      totalActiveUsers,
      revenueChart,
      growth,
    });
  } catch (error) {
    res.status(500).json({ success: false, message: (error as Error).message });
  }
};

// Task 1: Suspend a user's subscription
export const suspendUserSubscription = async (req: Request, res: Response): Promise<void> => {
  try {
    const { userId } = req.params;

    const subscription = await Subscription.findOne({ user: userId });
    if (!subscription) {
      res.status(404).json({ success: false, message: 'Subscription not found.' });
      return;
    }

    if (subscription.trialActive) {
      res.status(400).json({ success: false, message: 'Cannot suspend a subscription in trial period.' });
      return;
    }

    // Suspend by setting endDate to now
    subscription.endDate = new Date();
    await subscription.save();

    res.status(200).json({ success: true, message: 'Subscription suspended successfully.' });
  } catch (error) {
    res.status(500).json({ success: false, message: (error as Error).message });
  }
};


// Task 1: Delete a specific user
export const deleteUser = async (req: Request, res: Response): Promise<void> => {
  try {
    const { userId } = req.params;

    const user = await User.findById(userId);
    if (!user) {
      res.status(404).json({ success: false, message: 'User not found.' });
      return;
    }

    // Delete associated data
    await Subscription.deleteMany({ user: userId });
    await ChatHistory.deleteMany({ userId });

    // Delete the user
    await User.findByIdAndDelete(userId);

    res.status(200).json({ success: true, message: 'User deleted successfully.' });
  } catch (error) {
    res.status(500).json({ success: false, message: (error as Error).message });
  }
};


// Get user details with chat history
export const getUserDetails = async (req: Request, res: Response): Promise<void> => {
  try {
    const { userId } = req.params;

    const user = await User.findById(userId).select('-password');
    if (!user) {
      res.status(404).json({ success: false, message: 'User not found.' });
      return;
    }

    const chatHistory = await ChatHistory.find({ userId }).select('chat_name chat_contents timestamps');
    const subscription = await Subscription.findOne({ user: userId }).populate('package');

    res.status(200).json({
      success: true,
      user: {
        ...user.toObject(),
        subscription: subscription
          ? {
              subscriptionType: (subscription.package as any).subscriptionType,
              status: subscription.trialActive
                ? 'Trial'
                : subscription.endDate > new Date()
                ? 'Active'
                : 'Expired',
              amount: `$${(subscription.package as any).amount / 100}`,
              startDate: subscription.startDate,
              endDate: subscription.endDate,
              trialActive: subscription.trialActive,
            }
          : null,
      },
      chatHistory,
    });
  } catch (error) {
    res.status(500).json({ success: false, message: (error as Error).message });
  }
};

// Delete a conversation
export const deleteConversation = async (req: Request, res: Response): Promise<void> => {
  try {
    const { conversationId } = req.params;

    const conversation = await ChatHistory.findById(conversationId);
    if (!conversation) {
      res.status(404).json({ success: false, message: 'Conversation not found.' });
      return;
    }

    await ChatHistory.findByIdAndDelete(conversationId);

    res.status(200).json({ success: true, message: 'Conversation deleted successfully.' });
  } catch (error) {
    res.status(500).json({ success: false, message: (error as Error).message });
  }
};

export const uploadAHeroImage = async (req: Request, res: Response): Promise<void> => {
  try {
    if (!req.file) {
      res.status(400).json({ success: false, message: 'No file uploaded' });
      return;
    }
    const result = await uploadHeroImage(req.file.path);
    await fs.unlink(req.file.path);
    const heroImageUrl = result.secure_url;
    res.status(200).json({ success: true, heroImageUrl });
  } catch (error) {
    res.status(500).json({ success: false, message: (error as Error).message });
  }
};

// Task: Get all hero section images
export const getAllHeroImages = async (req: Request, res: Response): Promise<void> => {
  try {
    const heroImages = await getHeroImages();
    res.status(200).json({
      success: true,
      heroImages,
    });
  } catch (error) {
    console.error('Error in getAllHeroImages:', error);
    const message = error instanceof Error ? error.message : 'Failed to retrieve hero images: Unknown error';
    res.status(429).json({
      success: false,
      message,
    });
  }
};

// Task: Delete a hero image
export const deleteHeroImageController = async (req: Request, res: Response): Promise<void> => {
  try {
    const { publicId } = req.params;
    if (!publicId) {
      res.status(400).json({ success: false, message: 'Public ID is required' });
      return;
    }

    // Ensure the publicId includes the folder prefix if necessary
    const fullPublicId = publicId.startsWith('hero_images/') ? publicId : `hero_images/${publicId}`;
    await deleteHeroImage(fullPublicId);
    res.status(200).json({ success: true, message: `Hero image ${fullPublicId} deleted successfully` });
  } catch (error) {
    console.error('Error in deleteHeroImageController:', error);
    res.status(500).json({
      success: false,
      message: error instanceof Error ? error.message : 'Failed to delete hero image: Unknown error',
    });
  }
};