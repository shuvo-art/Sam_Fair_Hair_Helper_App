import { RequestHandler } from 'express';
import { Policy } from './policy.model';

// Create Policy
export const createPolicy: RequestHandler = async (req, res) => {
  try {
    const { type, content } = req.body;

    if (!type || !content) {
      res.status(400).json({ success: false, message: 'Type and content are required.' });
      return;
    }

    const existingPolicy = await Policy.findOne({ type });
    if (existingPolicy) {
      res.status(400).json({ success: false, message: 'Policy already exists.' });
      return;
    }

    const policy = new Policy({ type, content });
    await policy.save();

    res.status(201).json({ success: true, policy });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    res.status(500).json({ success: false, message: errorMessage });
  }
};

// Get Policy by Type
export const getPolicy: RequestHandler = async (req, res) => {
  try {
    const { type } = req.params;

    const policy = await Policy.findOne({ type });
    if (!policy) {
      res.status(404).json({ success: false, message: 'Policy not found.' });
      return;
    }

    res.status(200).json({ success: true, policy });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    res.status(500).json({ success: false, message: errorMessage });
  }
};

// Update Policy by Type
export const updatePolicy: RequestHandler = async (req, res) => {
  try {
    const { type } = req.params;
    const { content } = req.body;

    if (!content) {
      res.status(400).json({ success: false, message: 'Content is required.' });
      return;
    }

    const policy = await Policy.findOneAndUpdate(
      { type },
      { content },
      { new: true, upsert: true }
    );

    res.status(200).json({ success: true, policy });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    res.status(500).json({ success: false, message: errorMessage });
  }
};
