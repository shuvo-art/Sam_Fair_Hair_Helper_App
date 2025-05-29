import express, { Request, Response } from 'express';
import Stripe from 'stripe';
import { Subscription } from './subscription.model';
import { User } from '../user/user.model';
import { Package } from './package.model';

const router = express.Router();
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY as string, {
  apiVersion: '2024-12-18.acacia',
});

router.post('/webhook', express.raw({ type: 'application/json' }), async (req: Request, res: Response): Promise<void> => {
  try {
    const sig = req.headers['stripe-signature'];
    if (!sig) {
      throw new Error('Stripe-Signature header is missing');
    }

    const rawBody = req.body instanceof Buffer ? req.body : Buffer.from(req.body);
    const event = stripe.webhooks.constructEvent(rawBody, sig, process.env.STRIPE_WEBHOOK_SECRET as string);

    if (event.type === 'checkout.session.completed') {
      const session = event.data.object as Stripe.Checkout.Session;
      const userId = session.metadata?.userId;
      const packId = session.metadata?.packId;

      if (!userId || !packId) {
        throw new Error('User ID or Package ID is missing');
      }

      const selectedPackage = await Package.findOne({ packId, status: 'Active' });
      if (!selectedPackage) {
        throw new Error('Package not found or suspended');
      }

      const stripeSubscription = await stripe.subscriptions.retrieve(session.subscription as string);
      const stripeSubscriptionId = stripeSubscription.id;

      const startDate = new Date();
      const endDate = new Date();
      endDate.setMonth(startDate.getMonth() + (selectedPackage.subscriptionType === 'Yearly' ? 12 : 1));

      await Subscription.findOneAndUpdate(
        { user: userId },
        {
          user: userId,
          package: selectedPackage._id,
          startDate,
          endDate,
          trialActive: stripeSubscription.status === 'trialing',
          stripeSubscriptionId,
        },
        { new: true, upsert: true }
      );
    } else if (event.type === 'customer.subscription.updated') {
      const subscription = event.data.object as Stripe.Subscription;
      const dbSubscription = await Subscription.findOne({ stripeSubscriptionId: subscription.id });
      if (dbSubscription) {
        dbSubscription.trialActive = subscription.status === 'trialing';
        if (subscription.cancel_at_period_end) {
          dbSubscription.endDate = new Date(subscription.current_period_end * 1000);
        }
        await dbSubscription.save();
      }
    } else if (event.type === 'customer.subscription.deleted') {
      const subscription = event.data.object as Stripe.Subscription;
      const dbSubscription = await Subscription.findOne({ stripeSubscriptionId: subscription.id });
      if (dbSubscription) {
        await Subscription.deleteOne({ stripeSubscriptionId: subscription.id });
      }
    } else if (event.type === 'invoice.payment_succeeded') {
      const invoice = event.data.object as Stripe.Invoice;
      const subscriptionId = invoice.subscription as string;
      const amountPaid = invoice.amount_paid / 100;

      const subscription = await Subscription.findOne({ stripeSubscriptionId: subscriptionId }).populate('package');
      if (subscription && subscription.trialActive) {
        const stripeSubscription = await stripe.subscriptions.retrieve(subscriptionId);
        if (stripeSubscription.status !== 'trialing') {
          subscription.trialActive = false;
          const selectedPackage = subscription.package as any;
          const startDate = new Date();
          const endDate = new Date();
          endDate.setMonth(startDate.getMonth() + (selectedPackage.subscriptionType === 'Yearly' ? 12 : 1));
          subscription.startDate = startDate;
          subscription.endDate = endDate;
          subscription.amountPaid = amountPaid;
          await subscription.save();
        }
      }
    }

    res.status(200).json({ success: true, received: true });
  } catch (error: any) {
    console.error('Webhook error:', error.message);
    res.status(400).json({ success: false, message: error.message });
  }
});

export default router;