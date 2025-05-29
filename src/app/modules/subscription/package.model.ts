import mongoose, { Schema, Document } from 'mongoose';

export interface IPackage extends Document {
  packId: string; // Unique identifier (e.g., "MONTHLY_001")
  amount: number; // Amount in cents (e.g., 3200 for $32)
  subscriptionType: 'Monthly' | 'Yearly';
  status: 'Active' | 'Suspended';
  currency?: string; // Default to "USD"
  freeTrialDays: number; // Number of free trial days (e.g., 1 for one day)
  stripePriceId: string; // Stripe price ID
}

const PackageSchema: Schema = new Schema(
  {
    packId: { type: String, required: true, unique: true },
    amount: { type: Number, required: true },
    subscriptionType: { type: String, enum: ['Monthly', 'Yearly'], required: true },
    status: { type: String, enum: ['Active', 'Suspended'], default: 'Active' },
    currency: { type: String, default: 'USD' },
    freeTrialDays: { type: Number, default: 0 }, // Default to 0, set to 1 for trials
    stripePriceId: { type: String},
  },
  { timestamps: true }
);

export const Package = mongoose.model<IPackage>('Package', PackageSchema);