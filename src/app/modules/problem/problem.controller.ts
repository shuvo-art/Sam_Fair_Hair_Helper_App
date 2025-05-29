import { Request, Response } from 'express';
import nodemailer from 'nodemailer';

export const reportProblem = async (req: Request, res: Response): Promise<void> => {
  const { email, description } = req.body;
  console.log('Email:', req.body);

  if (!email || !description) {
    res.status(400).json({ success: false, message: 'Email and description are required.' });
    return;
  }

  try {
    const transporter = nodemailer.createTransport({
      service: 'gmail',
      auth: {
        user: process.env.EMAIL_USER, // Admin email used for SMTP
        pass: process.env.EMAIL_PASS,
      },
    });

    await transporter.sendMail({
      from: process.env.EMAIL_USER, // Admin email
      to: process.env.ADMIN_EMAIL, // Admin email receiving the message
      subject: 'User Reported Problem',
      text: `Email: ${email}\n\nDescription: ${description}`,
      replyTo: email, // User's email for replies
    });

    res.status(200).json({ success: true, message: 'Problem reported successfully.' });
  } catch (error) {
    console.error('Error sending email:', error);
    res.status(500).json({ success: false, message: 'Failed to report the problem.' });
  }
};
