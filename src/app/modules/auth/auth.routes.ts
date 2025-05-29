import express, { Request, Response, RequestHandler } from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { registerUser, loginUser, generateOTP, verifyOTP, generateAccessToken, generateRefreshToken, verifyRefreshToken, otpMap } from './auth.service';
import { User } from '../user/user.model';
import { Subscription } from '../subscription/subscription.model';
import { Package } from '../subscription/package.model';
import { authenticate, requireRole } from '../auth/auth.middleware';
import { z } from 'zod';

const router = express.Router();

// Validation schemas
const signupSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  name: z.string().min(1),
  birthday: z.string().optional(),
  fcmToken: z.string().optional(),
  phone: z.string().optional(), // Add phone
  country: z.string().optional(), // Add country
});

const loginSchema = z.object({
  email: z.string().email(),
  password: z.string(),
  isAdmin: z.boolean().optional(),
  fcmToken: z.string().optional(),
});

const oauthSchema = z.object({
  email: z.string().email(),
  name: z.string().min(1),
  profileImage: z.string().optional(),
  fcmToken: z.string().optional(),
  phone: z.string().optional(), // Add phone
  country: z.string().optional(), // Add country
});

const otpRequestSchema = z.object({
  email: z.string().email(),
  otp: z.string().optional(),
});

const resetPasswordSchema = z.object({
  email: z.string().email(),
  newPassword: z.string().min(8),
});

const refreshTokenSchema = z.object({
  token: z.string(),
});

let refreshTokens: string[] = [];
const otpCache = new Map<string, string>();

router.post("/check-email", async (req: Request, res: Response): Promise<void> => {
  const { email } = req.body;

  try {
    const user = await User.findOne({ email });
    res.status(200).json({ exists: !!user });
  } catch (error) {
    console.error("Error checking email availability:", error);
    res.status(500).json({ error: "An error occurred. Please try again later." });
  }
});

// Signup route
router.post('/signup', async (req: Request, res: Response): Promise<void> => {
  try {
    const { email, password, name, birthday, fcmToken, phone, country } = signupSchema.parse(req.body);
    const user = await registerUser(email, password, name, birthday ? new Date(birthday) : undefined, fcmToken, phone, country);

    const accessToken = generateAccessToken(user);
    const refreshToken = generateRefreshToken(user);

    refreshTokens.push(refreshToken);

    const otp = await generateOTP(email);

    const userResponse = {
      _id: user._id,
      email: user.email,
      name: user.name,
      role: user.role,
      language: user.language,
      birthday: user.birthday,
      fcmToken: user.fcmToken,
      phone: user.phone, // Include phone in response
      country: user.country, // Include country in response
    };

    res.status(201).json({
      success: true,
      message: 'User registered successfully. OTP sent to email.',
      otp, // Remove in production
      user: userResponse,
      accessToken,
      refreshToken,
    });
  } catch (error: any) {
    res.status(400).json({ success: false, message: error.message });
  }
});

// Login
router.post('/login', async (req: Request, res: Response): Promise<void> => {
  try {
    const { email, password, fcmToken } = loginSchema.parse(req.body);
    const user = await User.findOne({ email });

    if (!user || !(await bcrypt.compare(password, user.password))) {
      res.status(401).json({ success: false, message: 'Invalid email or password' });
      return;
    }

    // Update fcmToken if provided
    if (fcmToken) {
      user.fcmToken = fcmToken;
      await user.save();
    }

    const existingSubscription = await Subscription.findOne({ user: user._id });
    if (!existingSubscription) {
      let freePackage = await Package.findOne({ subscriptionType: 'Monthly', amount: 0, status: 'Active' });
      if (!freePackage) {
        freePackage = await new Package({
          packId: 'FREE_001',
          amount: 0,
          subscriptionType: 'Monthly',
          status: 'Active',
          currency: 'USD',
          freeTrialDays: 0,
        }).save();
      }

      const startDate = new Date();
      const endDate = new Date();
      endDate.setMonth(startDate.getMonth() + 1);

      await Subscription.create({
        user: user._id,
        package: freePackage._id,
        startDate,
        endDate,
        trialActive: false,
      });
    }

    const accessToken = generateAccessToken(user);
    const refreshToken = generateRefreshToken(user);
    refreshTokens.push(refreshToken);

    res.status(200).json({
      success: true,
      user: {
        _id: user._id,
        email: user.email,
        name: user.name,
        role: user.role,
        language: user.language,
        profileImage: user.profileImage,
        fcmToken: user.fcmToken,
        birthday: user.birthday,
        phone: user.phone, // Include phone in response
        country: user.country, // Include country in response
      },
      accessToken,
      refreshToken,
    });
  } catch (error: any) {
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    res.status(500).json({ success: false, message: errorMessage });
  }
});

// Google OAuth Login/Signup
router.post('/oauth/google', (async (req: Request, res: Response) => {
  try {
    const { email, name, profileImage, fcmToken, phone, country } = oauthSchema.parse(req.body);

    if (!email || !name) {
      return res.status(400).json({ success: false, message: 'Email and name are required' });
    }

    let user = await User.findOne({ email });

    if (!user) {
      user = new User({
        email,
        name,
        profileImage,
        password: 'oauth_temp_password',
        fcmToken,
        phone, // Set phone if provided
        country, // Set country if provided
      });
      await user.save();

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
        user: user._id,
        package: freePackage._id,
        startDate,
        endDate,
        trialActive: false,
      });
    } else {
      // Update fcmToken, phone, and country for existing user if provided
      if (fcmToken) user.fcmToken = fcmToken;
      if (phone) user.phone = phone;
      if (country) user.country = country;
      await user.save();
    }

    const accessToken = generateAccessToken(user);
    const refreshToken = generateRefreshToken(user);
    refreshTokens.push(refreshToken);

    res.status(200).json({
      success: true,
      user: {
        _id: user._id,
        email: user.email,
        name: user.name,
        role: user.role,
        profileImage: user.profileImage,
        language: user.language,
        fcmToken: user.fcmToken,
        birthday: user.birthday,
        phone: user.phone, // Include phone in response
        country: user.country, // Include country in response
      },
      accessToken,
      refreshToken,
    });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
}) as RequestHandler);

// Apple OAuth Login/Signup
router.post('/oauth/apple', (async (req: Request, res: Response) => {
  try {
    const { email, name, profileImage, fcmToken, phone, country } = oauthSchema.parse(req.body);

    if (!email || !name) {
      return res.status(400).json({ success: false, message: 'Email and name are required' });
    }

    let user = await User.findOne({ email });

    if (!user) {
      user = new User({
        email,
        name,
        profileImage,
        password: 'oauth_temp_password',
        fcmToken,
        phone, // Set phone if provided
        country, // Set country if provided
      });
      await user.save();

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
        user: user._id,
        package: freePackage._id,
        startDate,
        endDate,
        trialActive: false,
      });
    } else {
      // Update fcmToken, phone, and country for existing user if provided
      if (fcmToken) user.fcmToken = fcmToken;
      if (phone) user.phone = phone;
      if (country) user.country = country;
      await user.save();
    }

    const accessToken = generateAccessToken(user);
    const refreshToken = generateRefreshToken(user);
    refreshTokens.push(refreshToken);

    res.status(200).json({
      success: true,
      user: {
        _id: user._id,
        email: user.email,
        name: user.name,
        role: user.role,
        profileImage: user.profileImage,
        language: user.language,
        fcmToken: user.fcmToken,
        birthday: user.birthday,
        phone: user.phone, // Include phone in response
        country: user.country, // Include country in response
      },
      accessToken,
      refreshToken,
    });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
}) as RequestHandler);

// Send OTP for password reset
router.post(
  '/password/reset',
  (async (req: Request, res: Response): Promise<void> => {
    try {
      const { email } = otpRequestSchema.parse(req.body);
      const otp = await generateOTP(email);
      otpCache.set(email, otp);
      res.status(200).json({ success: true, message: 'OTP sent to email' });
    } catch (error: any) {
      res.status(500).json({ success: false, message: error.message });
    }
  }) as RequestHandler
);

// Verify OTP
router.post(
  '/verify-otp',
  (async (req: Request, res: Response): Promise<void> => {
    try {
      const { email, otp } = otpRequestSchema.parse(req.body);
      const cachedOTP = otpMap.get(email);

      if (!cachedOTP || cachedOTP !== otp) {
        res.status(400).json({ success: false, message: 'Invalid OTP' });
        return;
      }

      otpMap.delete(email);
      res.status(200).json({ success: true, message: 'OTP verified successfully' });
    } catch (error: any) {
      res.status(400).json({ success: false, message: error.message });
    }
  }) as RequestHandler
);

// Verify OTP and reset password
router.post(
  '/password/reset/verify',
  (async (req: Request, res: Response): Promise<void> => {
    try {
      const { email, newPassword } = resetPasswordSchema.parse(req.body);

      const hashedPassword = await bcrypt.hash(newPassword, 10);
      await User.findOneAndUpdate({ email }, { password: hashedPassword });
      res.status(200).json({ success: true, message: 'Password reset successful' });
    } catch (error: any) {
      res.status(500).json({ success: false, message: error.message });
    }
  }) as RequestHandler
);

// Refresh token route
router.post(
  '/refresh-token',
  (async (req: Request, res: Response): Promise<void> => {
    try {
      const { token } = refreshTokenSchema.parse(req.body);

      if (!refreshTokens.includes(token)) {
        res.status(403).json({ success: false, message: 'Refresh token is invalid' });
        return;
      }

      const decoded = jwt.verify(token, process.env.REFRESH_TOKEN_SECRET as string) as { id: string };
      const user = await User.findById(decoded.id);

      if (!user) {
        res.status(404).json({ success: false, message: 'User not found' });
        return;
      }

      const accessToken = generateAccessToken(user);

      res.status(200).json({ success: true, accessToken });
    } catch (error: any) {
      res.status(401).json({ success: false, message: 'Invalid or expired refresh token' });
    }
  }) as RequestHandler
);

// Logout route
router.post('/logout', (req: Request, res: Response): void => {
  const { token } = req.body;

  refreshTokens = refreshTokens.filter((t) => t !== token);
  res.status(200).json({ success: true, message: 'Logged out successfully' });
});

export default router;