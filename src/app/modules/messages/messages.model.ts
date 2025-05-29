import mongoose, { Schema, Document } from 'mongoose';

export interface IMessage extends Document {
  chat: mongoose.Types.ObjectId;
  sender: mongoose.Types.ObjectId | 'bot';
  content: string;
}

const MessageSchema: Schema = new Schema(
  {
    chat: { type: mongoose.Types.ObjectId, ref: 'Chat', required: true },
    sender: { type: mongoose.Types.ObjectId, ref: 'User', required: true },
    content: { type: String, required: true },
  },
  { timestamps: true }
);

export const Message = mongoose.model<IMessage>('Message', MessageSchema);
 


