import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import crypto from 'crypto';
import nodemailer from 'nodemailer';
import { User, IUser } from '../user/user.model';
import { Subscription } from '../subscription/subscription.model';
import { Package } from '../subscription/package.model';

export const registerUser = async (email: string, password: string, name: string, birthday?: Date, fcmToken?: string, phone?: string, country?: string) => {
  const hashedPassword = await bcrypt.hash(password, 10);
  const newUser = new User({
    email,
    password: hashedPassword,
    name,
    birthday,
    fcmToken,
    phone, // Add phone to new user
    country, // Add country to new user
  });
  const savedUser = await newUser.save();

  let freePackage = await Package.findOne({ subscriptionType: 'Monthly', amount: 0, status: 'Active' });
  if (!freePackage) {
    freePackage = new Package({
      packId: 'FREE_001',
      amount: 0,
      subscriptionType: 'Monthly',
      status: 'Active',
      currency: 'USD',
      freeTrialDays: 0,
    });
    await freePackage.save();
  }

  const startDate = new Date();
  const endDate = new Date();
  endDate.setMonth(startDate.getMonth() + 1);

  await Subscription.create({
    user: savedUser._id,
    package: freePackage._id,
    startDate,
    endDate,
    trialActive: false,
  });

  return savedUser;
};

export const loginUser = async (email: string, password: string) => {
  const user = await User.findOne({ email });
  if (!user || !(await bcrypt.compare(password, user.password))) {
    throw new Error('Invalid email or password');
  }
  const token = jwt.sign({ id: user._id, role: user.role }, process.env.JWT_SECRET as string, {
    expiresIn: '7d',
  });
  return { user, token, role: user.role };
};

export const otpMap = new Map<string, string>();

export const generateOTP = async (email: string) => {
  const otp = crypto.randomInt(1000, 9999).toString();
  otpMap.set(email, otp);

  const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS,
    },
  });

  const mailOptions = {
    from: `"Your App Name" <${process.env.EMAIL_USER}>`,
    to: email,
    subject: 'Your One-Time Password (OTP) for Verification',
    html: `
      <div style="font-family: Arial, sans-serif; color: #333;">
        <h2 style="color: #4CAF50;">Your OTP for Verification</h2>
        <p>Hello,</p>
        <p>Your One-Time Password (OTP) for verification is:</p>
        <h3 style="background: #f4f4f4; padding: 10px; display: inline-block; border-radius: 5px;">${otp}</h3>
        <p>This OTP is valid for a limited time. Please do not share it with anyone.</p>
        <p>If you did not request this OTP, please ignore this email.</p>
        <hr style="border: 1px solid #ddd;">
        <p style="font-size: 12px; color: #777;">
          This email was sent by <strong>Your App Name</strong>. 
          If you have any questions, please contact us at <a href="mailto:support@yourapp.com">support@yourapp.com</a>.
        </p>
      </div>
    `,
    headers: {
      'Reply-To': process.env.EMAIL_USER || '',
      'List-Unsubscribe': `<mailto:unsubscribe@yourapp.com?subject=Unsubscribe>`,
    },
  };

  try {
    await transporter.sendMail(mailOptions);
    console.log(`OTP sent to email: ${email}`);
  } catch (error) {
    console.error(`Error sending OTP to ${email}:`, error);
    throw new Error('Failed to send OTP. Please try again.');
  }

  return otp;
};

export const verifyOTP = (email: string, otp: string) => {
  const validOTP = otpMap.get(email);
  if (validOTP === otp) {
    otpMap.delete(email);
    return true;
  }
  return false;
};

const refreshTokens: string[] = [];

export const generateAccessToken = (user: IUser) => {
  return jwt.sign({ id: user._id, role: user.role }, process.env.JWT_SECRET as string, { expiresIn: '7d' });
};

export const generateRefreshToken = (user: IUser) => {
  const refreshToken = jwt.sign({ id: user._id }, process.env.REFRESH_TOKEN_SECRET as string, { expiresIn: '7d' });
  refreshTokens.push(refreshToken);
  return refreshToken;
};

export const verifyRefreshToken = (token: string) => {
  try {
    return jwt.verify(token, process.env.REFRESH_TOKEN_SECRET as string);
  } catch {
    return null;
  }
};