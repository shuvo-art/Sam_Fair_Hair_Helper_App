import { RequestHandler } from 'express';
import jwt from 'jsonwebtoken';
import { User } from '../user/user.model';

// Middleware to authenticate users and attach the user object to the request
export const authenticate: RequestHandler = async (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) {
    res.status(401).json({ success: false, message: 'Unauthorized' });
    return;
  }

  try {
    const decoded: any = jwt.verify(token, process.env.JWT_SECRET as string);
    const user = await User.findById(decoded.id);
    if (!user) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }
    req.user = user; // Attach authenticated user to the request object
    req.user.role = decoded.role; // Attach role from the token
    next();
  } catch (error) {
    res.status(401).json({ success: false, message: 'Invalid or expired token' });
  }
};

// Middleware to enforce role-based access control
export const requireRole = (role: string): RequestHandler => {
  return (req, res, next) => {
    if (!req.user || req.user.role !== role) {
      res.status(403).json({ success: false, message: 'Forbidden: Access is denied' });
      return;
    }
    next();
  };
};
