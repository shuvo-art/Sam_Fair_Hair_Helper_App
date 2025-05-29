import mongoose, { Schema, Document } from 'mongoose';

export interface ISubscription extends Document {
  user: mongoose.Types.ObjectId;
  package: mongoose.Types.ObjectId;
  startDate: Date;
  endDate: Date;
  trialActive: boolean;
  stripeSubscriptionId?: string;
  amountPaid?: number; // New field to store the paid amount from Stripe
}

const SubscriptionSchema: Schema = new Schema(
  {
    user: { type: mongoose.Types.ObjectId, ref: 'User', required: true },
    package: { type: mongoose.Types.ObjectId, ref: 'Package', required: true },
    startDate: { type: Date, required: true },
    endDate: { type: Date, required: true },
    trialActive: { type: Boolean, default: false },
    stripeSubscriptionId: { type: String, default: null },
    amountPaid: { type: Number, default: null }, // Optional field for payment tracking
  },
  { timestamps: true }
);

export const Subscription = mongoose.model<ISubscription>('Subscription', SubscriptionSchema);