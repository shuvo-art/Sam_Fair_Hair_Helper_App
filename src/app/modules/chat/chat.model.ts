import mongoose, { Schema, Document } from 'mongoose';

export interface IChat extends Document {
  participants: mongoose.Types.ObjectId[];
}

const ChatSchema: Schema = new Schema(
  {
    participants: [{ type: mongoose.Types.ObjectId, ref: 'User', required: true }],
  },
  { timestamps: true }
);

export const Chat = mongoose.model<IChat>('Chat', ChatSchema);
