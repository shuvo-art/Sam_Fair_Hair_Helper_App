import express, { Request, Response } from 'express';
import { Package } from './package.model';
import { authenticate, requireRole } from '../auth/auth.middleware';
import { z } from 'zod';
import Stripe from 'stripe';
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY as string, {
  apiVersion: '2024-12-18.acacia',
});

const router = express.Router();

// Validation schema for adding/updating a package
const packageSchema = z.object({
  packId: z.string().min(1),
  amount: z.number().positive().optional(),
  subscriptionType: z.enum(['Monthly', 'Yearly']),
  currency: z.string().optional(),
  freeTrialDays: z.number().min(0).optional(),
  stripePriceId: z.string().optional(),
  status: z.string().optional(),
});

// Add a new package (Task 1)
router.post(
  '/add',
  authenticate,
  requireRole('admin'),
  async (req: Request, res: Response): Promise<void> => {
    try {
      const { packId, amount, subscriptionType, currency, freeTrialDays, stripePriceId } = req.body;

      if (!stripePriceId || !stripePriceId.startsWith('price_')) {
        res.status(400).json({ success: false, message: 'Invalid or missing Stripe Price ID.' });
        return;
      }

      const existingPackage = await Package.findOne({ packId });
      if (existingPackage) {
        res.status(400).json({ success: false, message: 'Package with this ID already exists' });
        return;
      }

      // Fetch the price from Stripe to ensure consistency
      const stripePrice = await stripe.prices.retrieve(stripePriceId);
      const syncedAmount = stripePrice.unit_amount || amount || 0;

      const newPackage = new Package({
        packId,
        amount: syncedAmount,
        subscriptionType,
        status: 'Active',
        currency: currency || 'USD',
        freeTrialDays: freeTrialDays || 0,
        stripePriceId,
      });

      await newPackage.save();

      res.status(201).json({ success: true, package: newPackage });
    } catch (error: any) {
      res.status(500).json({ success: false, message: error.message });
    }
  }
);

// Update a package (Task 2)
router.put(
  '/:packId',
  authenticate,
  requireRole('admin'),
  async (req: Request, res: Response): Promise<void> => {
    try {
      const { packId } = req.params;
      const { amount, status, currency, freeTrialDays, stripePriceId } = packageSchema.partial().parse(req.body);

      const packageToUpdate = await Package.findOne({ packId });
      if (!packageToUpdate) {
        res.status(404).json({ success: false, message: 'Package not found' });
        return;
      }

      // Initialize variables
      let syncedAmount: number = packageToUpdate.amount;
      let newStripePriceId: string | undefined = stripePriceId || packageToUpdate.stripePriceId;

      // If a new stripePriceId is provided, fetch its amount
      if (stripePriceId && stripePriceId !== packageToUpdate.stripePriceId) {
        const stripePrice = await stripe.prices.retrieve(stripePriceId);
        syncedAmount = amount !== undefined ? amount : stripePrice.unit_amount ?? packageToUpdate.amount;
        newStripePriceId = stripePriceId;
      } else if (amount !== undefined && amount !== packageToUpdate.amount) {
        // If the amount has changed, we need to create a new Stripe Price
        if (!packageToUpdate.stripePriceId) {
          res.status(400).json({ success: false, message: 'Package does not have a Stripe Price ID to update.' });
          return;
        }

        // Fetch the existing Stripe Price to get the Product ID
        const existingStripePrice = await stripe.prices.retrieve(packageToUpdate.stripePriceId);
        const productId = existingStripePrice.product as string;

        // Create a new Stripe Price with the updated amount
        const newStripePrice = await stripe.prices.create({
          product: productId,
          unit_amount: amount,
          currency: packageToUpdate.currency || 'USD',
          recurring: {
            interval: packageToUpdate.subscriptionType.toLowerCase() as 'month' | 'year',
          },
        });

        // Update the Product to set the new Price as the default
        await stripe.products.update(productId, {
          default_price: newStripePrice.id,
        });

        // Update the synced amount and stripePriceId
        syncedAmount = amount;
        newStripePriceId = newStripePrice.id;
      }

      // Update the package in the database
      const updatedPackage = await Package.findOneAndUpdate(
        { packId },
        { amount: syncedAmount, status, currency, freeTrialDays, stripePriceId: newStripePriceId },
        { new: true }
      );

      if (!updatedPackage) {
        res.status(404).json({ success: false, message: 'Package not found' });
        return;
      }

      res.status(200).json({ success: true, package: updatedPackage });
    } catch (error: any) {
      res.status(500).json({ success: false, message: error.message });
    }
  }
);

// Suspend a package (Task 2)
router.put(
  '/:packId/suspend',
  authenticate,
  requireRole('admin'),
  async (req: Request, res: Response): Promise<void> => {
    try {
      const { packId } = req.params;

      const packageToSuspend = await Package.findOneAndUpdate(
        { packId },
        { status: 'Suspended' },
        { new: true }
      );

      if (!packageToSuspend) {
        res.status(404).json({ success: false, message: 'Package not found' });
        return;
      }

      res.status(200).json({ success: true, package: packageToSuspend });
    } catch (error: any) {
      res.status(500).json({ success: false, message: error.message });
    }
  }
);

// Get all active packages (Task 5)
router.get(
  '/active',
  async (req: Request, res: Response): Promise<void> => {
    try {
      const activePackages = await Package.find({ status: 'Active' });
      res.status(200).json({ success: true, packages: activePackages });
    } catch (error: any) {
      res.status(500).json({ success: false, message: error.message });
    }
  }
);

// Get all packages for admin dashboard (Task 2)
router.get(
  '/',
  authenticate,
  requireRole('admin'),
  async (req: Request, res: Response): Promise<void> => {
    try {
      const packages = await Package.find();
      res.status(200).json({ success: true, packages });
    } catch (error: any) {
      res.status(500).json({ success: false, message: error.message });
    }
  }
);

export default router;