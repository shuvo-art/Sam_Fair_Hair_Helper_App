import jwt from 'jsonwebtoken';
import { User } from '../modules/user/user.model';
import { IUser } from '../modules/user/user.model';

export const getUserFromToken = async (token: string): Promise<IUser | null> => {
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET as string) as { id: string };
    return await User.findById(decoded.id) as IUser | null;
  } catch {
    return null;
  }
};
