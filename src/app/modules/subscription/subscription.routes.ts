import express, { Request, Response, RequestHandler } from 'express';
import Stripe from 'stripe';
import { Subscription, ISubscription } from './subscription.model';
import { User } from '../user/user.model';
import { Package } from './package.model';
import jwt from 'jsonwebtoken';
import { authenticate } from '../auth/auth.middleware';
import { createSubscriptionNotification } from '../../utils/notification.helper';

const router = express.Router();
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY as string, {
  apiVersion: '2024-12-18.acacia',
});

interface Params {
  userId: string;
}

// Initialize Free Subscription
router.post('/initialize', authenticate, async (req: Request, res: Response): Promise<void> => {
  try {
    const userId = req.user?._id;
    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized: Missing user ID.' });
      return;
    }

    const existingSubscription = await Subscription.findOne({ user: userId });
    if (existingSubscription) {
      res.status(400).json({ success: false, message: 'User already has a subscription.' });
      return;
    }

    const user = await User.findById(userId);
    if (!user) {
      res.status(404).json({ success: false, message: 'User not found.' });
      return;
    }

    const freePackage = await Package.findOne({ subscriptionType: 'Monthly', status: 'Active', amount: 0 });
    if (!freePackage) {
      res.status(500).json({ success: false, message: 'No active free package available.' });
      return;
    }

    const startDate = new Date();
    const endDate = new Date();
    endDate.setMonth(startDate.getMonth() + 1);

    const subscription = new Subscription({
      user: userId,
      package: freePackage._id,
      startDate,
      endDate,
      trialActive: false,
    });

    await subscription.save();

    res.status(201).json({ success: true, subscription });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
});

// Create Stripe session with one-day free trial
router.post('/stripe-session', authenticate, async (req: Request, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    const { packId } = req.body;

    if (!userId) {
      res.status(400).json({ success: false, message: 'User ID is missing.' });
      return;
    }

    const selectedPackage = await Package.findOne({ packId, status: 'Active' });
    if (!selectedPackage) {
      res.status(404).json({ success: false, message: 'Package not found or suspended.' });
      return;
    }

    if (!selectedPackage.stripePriceId || !selectedPackage.stripePriceId.startsWith('price_')) {
      res.status(500).json({ success: false, message: 'Package does not have a valid Stripe Price ID.' });
      return;
    }

    if (!process.env.ALLOWED_ORIGINS) {
      res.status(500).json({ success: false, message: 'Server configuration error: BASE_URL is not set.' });
      return;
    }

    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price: selectedPackage.stripePriceId,
          quantity: 1,
        },
      ],
      mode: 'subscription',
      subscription_data: {
        trial_period_days: selectedPackage.freeTrialDays || 0,
      },
      billing_address_collection: 'auto',
      success_url: `${process.env.ALLOWED_ORIGINS}/api/subscription/stripe-success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.ALLOWED_ORIGINS}/api/subscription/stripe-cancel`,
      metadata: { userId: String(userId), packId: selectedPackage.packId },
    });

    res.status(200).json({ success: true, sessionId: session.id, url: session.url });
  } catch (error: any) {
    console.error('Stripe session error:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
});

// Stripe success handler
router.get('/stripe-success', async (req: Request, res: Response): Promise<void> => {
  try {
    const { session_id } = req.query;
    const session = await stripe.checkout.sessions.retrieve(session_id as string, {
      expand: ['subscription'],
    });

    const userId = session.metadata?.userId;
    const packId = session.metadata?.packId;
    if (!userId || !packId) {
      res.status(400).json({ success: false, message: 'User ID or Package ID not found.' });
      return;
    }

    const token = req.headers.authorization?.split(' ')[1];
    if (!token) {
      res.status(401).json({ success: false, message: 'No token provided.' });
      return;
    }

    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    if (decoded.userId !== userId) {
      res.status(403).json({ success: false, message: 'Unauthorized access.' });
      return;
    }

    const selectedPackage = await Package.findOne({ packId, status: 'Active' });
    if (!selectedPackage) {
      res.status(404).json({ success: false, message: 'Package not found or suspended.' });
      return;
    }

    const stripeSubscription = session.subscription as Stripe.Subscription;
    const stripeSubscriptionId = stripeSubscription?.id;

    const startDate = new Date();
    const endDate = new Date();
    endDate.setMonth(startDate.getMonth() + (selectedPackage.subscriptionType === 'Yearly' ? 12 : 1));

    const isTrialActive = stripeSubscription.status === 'trialing';

    await Subscription.findOneAndUpdate(
      { user: userId },
      {
        user: userId,
        package: selectedPackage._id,
        startDate,
        endDate,
        trialActive: isTrialActive,
        stripeSubscriptionId,
      },
      { upsert: true, new: true }
    );

    res.status(200).json({ success: true, message: `Subscription started${isTrialActive ? ' with trial period' : ''}!` });
  } catch (error: any) {
    console.error('Stripe success error:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
});

// Cancel subscription
router.post('/cancel', authenticate, async (req: Request, res: Response): Promise<void> => {
  try {
    console.log('Step 1: Starting /cancel route');
    const userId = req.user?.id;
    console.log('Step 2: Extracted userId:', userId);
    if (!userId) {
      console.log('Step 3: No userId found, returning 401');
      res.status(401).json({ success: false, message: 'Unauthorized: No user ID.' });
      return;
    }

    console.log('Step 4: Querying subscription for userId:', userId);
    const subscription = await Subscription.findOne({ user: userId }).populate('package') as ISubscription & { _id: string };
    console.log('Step 5: Subscription found:', subscription);
    if (!subscription) {
      console.log('Step 6: No subscription found, returning 404');
      res.status(404).json({ success: false, message: 'No active subscription found.' });
      return;
    }

    console.log('Step 7: Checking stripeSubscriptionId:', subscription.stripeSubscriptionId);
    if (!subscription.stripeSubscriptionId) {
      console.log('Step 8: No stripeSubscriptionId found, deleting subscription from database');
      await Subscription.deleteOne({ user: userId });
      await createSubscriptionNotification(userId, 'subscription-canceled');
      console.log('Step 9: Subscription deleted from database');
      res.status(200).json({ success: true, message: 'Subscription canceled successfully (no Stripe subscription associated).' });
      return;
    }

    console.log('Step 9: Retrieving Stripe subscription for ID:', subscription.stripeSubscriptionId);
    const stripeSubscription = await stripe.subscriptions.retrieve(subscription.stripeSubscriptionId);
    console.log('Step 10: Stripe subscription retrieved:', stripeSubscription);
    const selectedPackage = subscription.package as any;
    console.log('Step 11: Selected package:', selectedPackage);

    if (stripeSubscription.status === 'trialing') {
      console.log('Step 12: Subscription is in trialing status, canceling immediately');
      await stripe.subscriptions.cancel(subscription.stripeSubscriptionId);
      console.log('Step 13: Stripe subscription canceled');
      await Subscription.deleteOne({ user: userId });
      console.log('Step 14: Subscription deleted from database');
      res.status(200).json({ success: true, message: 'Trial subscription canceled successfully.' });
    } else if (stripeSubscription.status === 'active') {
      console.log('Step 12: Subscription is active, scheduling cancellation');
      const updatedStripeSubscription = await stripe.subscriptions.update(subscription.stripeSubscriptionId, {
        cancel_at_period_end: true,
      });
      console.log('Step 13: Stripe subscription updated to cancel at period end:', updatedStripeSubscription);

      subscription.endDate = new Date(updatedStripeSubscription.current_period_end * 1000);
      console.log('Step 14: Updated subscription endDate:', subscription.endDate);
      await subscription.save();
      console.log('Step 15: Subscription saved in database');

      const period = selectedPackage.subscriptionType === 'Yearly' ? 'year' : 'month';
      res.status(200).json({
        success: true,
        message: `Subscription will be deactivated at the end of the current ${period} on ${subscription.endDate.toISOString().split('T')[0]}.`,
      });
    } else {
      console.log('Step 12: Subscription is not in a valid state for cancellation:', stripeSubscription.status);
      res.status(400).json({ success: false, message: 'Cannot cancel: Subscription is not in a valid state.' });
    }
  } catch (error: any) {
    console.error('Cancel error:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
});

// Renew subscription
router.put('/renew', authenticate, async (req: Request, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized: No user ID.' });
      return;
    }

    const subscription = await Subscription.findOne({ user: userId }).populate('package');
    if (!subscription) {
      res.status(404).json({ success: false, message: 'Subscription not found.' });
      return;
    }

    const currentDate = new Date();
    if (subscription.endDate > currentDate) {
      res.status(400).json({ success: false, message: 'Subscription is still active.' });
      return;
    }

    const selectedPackage = subscription.package as any;
    const startDate = new Date();
    const endDate = new Date();
    endDate.setMonth(startDate.getMonth() + (selectedPackage.subscriptionType === 'Yearly' ? 12 : 1));

    subscription.startDate = startDate;
    subscription.endDate = endDate;
    subscription.trialActive = false;
    await subscription.save();

    res.status(200).json({ success: true, subscription });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
});

// Get subscription details
router.get('/details', authenticate, async (req: Request, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized: No user ID.' });
      return;
    }

    let subscription = await Subscription.findOne({ user: userId }).populate('package');

    // If the subscription is missing (e.g., after cancellation), respond with a default inactive free trial
    if (!subscription) {
      const freePackage = await Package.findOne({ subscriptionType: 'Monthly', status: 'Active', amount: 0 });
      if (!freePackage) {
        res.status(404).json({ success: false, message: 'No subscription or free package found for this user.' });
        return;
      }

      const today = new Date();
      const startDate = new Date(today.setHours(0, 0, 0, 0));
      const endDate = new Date(startDate);
      endDate.setMonth(endDate.getMonth() + 1);

      res.status(200).json({
        success: true,
        subscription: {
          amount: "$0.00",
          begins: startDate.toISOString().split('T')[0],
          ends: endDate.toISOString().split('T')[0],
          type: 'Monthly',
          trialActive: false,
          trialExpires: null,
        },
      });
      return;
    }

    // Check if package is populated correctly
    const selectedPackage = subscription.package as any;
    if (!selectedPackage || !selectedPackage.amount === undefined) {
      console.error('Package not populated or missing amount for subscription:', subscription._id);
      res.status(500).json({ success: false, message: 'Subscription package data is incomplete.' });
      return;
    }

    const trialExpires =
      subscription.trialActive && selectedPackage.freeTrialDays
        ? new Date(subscription.startDate.getTime() + selectedPackage.freeTrialDays * 24 * 60 * 60 * 1000)
        : null;

    res.status(200).json({
      success: true,
      subscription: {
        amount: `$${(selectedPackage.amount / 100).toFixed(2)}`,
        begins: subscription.startDate.toISOString().split('T')[0],
        ends: subscription.endDate.toISOString().split('T')[0],
        type: selectedPackage.subscriptionType,
        trialActive: subscription.trialActive,
        trialExpires: trialExpires ? trialExpires.toISOString().split('T')[0] : null,
      },
    });
  } catch (error: any) {
    console.error('Error fetching subscription:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
});


export default router;