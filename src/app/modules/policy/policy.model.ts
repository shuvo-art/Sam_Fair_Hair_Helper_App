import mongoose, { Schema, Document } from 'mongoose';

export interface IPolicy extends Document {
  type: string; // 'privacy' or 'terms'
  content: string;
}

const PolicySchema: Schema = new Schema(
  {
    type: { type: String, enum: ['privacy', 'terms'], required: true, unique: true },
    content: { type: String, required: true },
  },
  { timestamps: true }
);

export const Policy = mongoose.model<IPolicy>('Policy', PolicySchema);
