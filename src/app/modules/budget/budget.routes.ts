import express, { Request, Response } from 'express';
import { Budget } from './budget.model';
import { authenticate } from '../auth/auth.middleware';
import { z } from 'zod';

const router = express.Router();

// Validation schema for a single budget entry
const budgetEntrySchema = z.object({
  category: z.enum(['Core Supports', 'Capacity Building Supports', 'Capital Supports']),
  subcategory: z.enum([
    'Assistance with daily life',
    'Assistance with social, economic, and community participation',
    'Consumables',
    'Transport',
    'Support coordination',
    'Improved living arrangements',
    'Increased social and community participation',
    'Finding and keeping a job',
    'Improved relationships',
    'Improved health and well-being',
    'Improved learning',
    'Improved life choices',
    'Improved daily living',
    'Assistive technology',
    'Equipment',
    'Vehicle modifications',
    'Home modifications',
    'Specialist disability accommodation'
  ]),
  amount: z.number().min(0)
});

// Validation schema for creating a budget
const createBudgetSchema = z.object({
  entries: z.array(budgetEntrySchema).min(1, 'At least one budget entry is required'),
  startDate: z.string().transform((val) => new Date(val)),
  endDate: z.string().transform((val) => new Date(val)),
});

// Validation schema for updating a budget
const updateBudgetSchema = z.object({
  entries: z.array(budgetEntrySchema).optional(),
  startDate: z.string().transform((val) => new Date(val)).optional(),
  endDate: z.string().transform((val) => new Date(val)).optional(),
});

// Validation schema for recording used budget amounts
const useBudgetSchema = z.object({
  entries: z.array(
    z.object({
      category: z.enum(['Core Supports', 'Capacity Building Supports', 'Capital Supports']),
      subcategory: z.enum([
        'Assistance with daily life',
        'Assistance with social, economic, and community participation',
        'Consumables',
        'Transport',
        'Support coordination',
        'Improved living arrangements',
        'Increased social and community participation',
        'Finding and keeping a job',
        'Improved relationships',
        'Improved health and well-being',
        'Improved learning',
        'Improved life choices',
        'Improved daily living',
        'Assistive technology',
        'Equipment',
        'Vehicle modifications',
        'Home modifications',
        'Specialist disability accommodation'
      ]),
      usedAmount: z.number().min(0)
    })
  ).min(1, 'At least one entry is required')
});

// Create budget with multiple entries
router.post('/', authenticate, async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    const parsedData = createBudgetSchema.parse(req.body);
    const { entries, startDate, endDate } = parsedData;

    // Check if a budget already exists for this user within the date range
    const budget = await Budget.findOne({
      userId,
      startDate: { $lte: endDate },
      endDate: { $gte: startDate }
    });

    if (budget) {
      // Update existing budget by merging entries
      entries.forEach((newEntry: any) => {
        const existingEntryIndex = budget.entries.findIndex(
          (entry: any) => entry.category === newEntry.category && entry.subcategory === newEntry.subcategory
        );
        if (existingEntryIndex !== -1) {
          // Update existing entry
          budget.entries[existingEntryIndex].amount = newEntry.amount;
        } else {
          // Add new entry
          budget.entries.push(newEntry);
        }
      });
      budget.startDate = startDate;
      budget.endDate = endDate;
      await budget.save();
      res.status(201).json({ success: true, budget });
    } else {
      // Create new budget
      const newBudget = new Budget({
        userId,
        entries,
        startDate,
        endDate,
      });
      await newBudget.save();
      res.status(201).json({ success: true, budget: newBudget });
    }
  } catch (error: any) {
    res.status(400).json({ success: false, message: error.message });
  }
});

// Update specific budget
router.put('/:id', authenticate, async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    const budgetId = req.params.id;

    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    const parsedData = updateBudgetSchema.parse(req.body);
    const budget = await Budget.findOne({ _id: budgetId, userId });

    if (!budget) {
      res.status(404).json({ success: false, message: 'Budget not found' });
      return;
    }

    // Update startDate and endDate if provided
    if (parsedData.startDate) budget.startDate = parsedData.startDate;
    if (parsedData.endDate) budget.endDate = parsedData.endDate;

    // Merge entries if provided
    if (parsedData.entries) {
      parsedData.entries.forEach((newEntry: any) => {
        const existingEntryIndex = budget.entries.findIndex(
          (entry: any) => entry.category === newEntry.category && entry.subcategory === newEntry.subcategory
        );
        if (existingEntryIndex !== -1) {
          // Update existing entry
          budget.entries[existingEntryIndex].amount = newEntry.amount;
        } else {
          // Add new entry
          budget.entries.push(newEntry);
        }
      });
    }

    await budget.save();
    res.status(200).json({ success: true, budget });
  } catch (error: any) {
    res.status(400).json({ success: false, message: error.message });
  }
});

// Record used budget amounts
router.post('/:id/use', authenticate, async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    const budgetId = req.params.id;

    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    const parsedData = useBudgetSchema.parse(req.body);
    const budget = await Budget.findOne({ _id: budgetId, userId });

    if (!budget) {
      res.status(404).json({ success: false, message: 'Budget not found' });
      return;
    }

    // Update usedAmount for specified entries
    parsedData.entries.forEach((newEntry: any) => {
      const existingEntryIndex = budget.entries.findIndex(
        (entry: any) => entry.category === newEntry.category && entry.subcategory === newEntry.subcategory
      );
      if (existingEntryIndex !== -1) {
        // Update usedAmount, ensuring it does not exceed the allocated amount
        const entry = budget.entries[existingEntryIndex];
        const newUsedAmount = entry.usedAmount + newEntry.usedAmount;
        if (newUsedAmount > entry.amount) {
          throw new Error(`Used amount for ${newEntry.subcategory} exceeds allocated amount of ${entry.amount}`);
        }
        budget.entries[existingEntryIndex].usedAmount = newUsedAmount;
      } else {
        throw new Error(`Entry with category ${newEntry.category} and subcategory ${newEntry.subcategory} not found`);
      }
    });

    await budget.save();

    // Calculate totals
    const totalAvailable = budget.entries.reduce((sum, entry) => sum + entry.amount, 0);
    const totalUsed = budget.entries.reduce((sum, entry) => sum + entry.usedAmount, 0);
    const remainingAmount = totalAvailable - totalUsed;

    res.status(200).json({
      success: true,
      budget,
      summary: {
        totalAvailable,
        totalUsed,
        remainingAmount
      }
    });
  } catch (error: any) {
    res.status(400).json({ success: false, message: error.message });
  }
});

// Get all budgets for the user
router.get('/', authenticate, async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    const budgets = await Budget.find({ userId });
    res.status(200).json({ success: true, budgets });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
});

// Get budget overview with percentages
router.get('/overview', authenticate, async (req: Request, res: Response) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    const budgets = await Budget.find({ userId });

    // Group budgets by category
    const categories = ['Core Supports', 'Capacity Building Supports', 'Capital Supports'];
    const overview = categories.reduce((acc, category) => {
      acc[category] = {
        total: 0,
        totalUsed: 0,
        remaining: 0,
        subcategories: {},
      };
      return acc;
    }, {} as Record<string, { total: number; totalUsed: number; remaining: number; subcategories: Record<string, { amount: number; usedAmount: number; percentage: number; remaining: number }> }>);

    // Calculate totals, used amounts, and percentages
    budgets.forEach((budget) => {
      budget.entries.forEach((entry: any) => {
        const category = entry.category;
        const subcategory = entry.subcategory;
        const amount = entry.amount;
        const usedAmount = entry.usedAmount;

        overview[category].total += amount;
        overview[category].totalUsed += usedAmount;
        overview[category].subcategories[subcategory] = {
          amount,
          usedAmount,
          percentage: 0, // Will calculate later
          remaining: amount - usedAmount
        };
      });
    });

    // Calculate percentages and remaining amounts within each category
    Object.keys(overview).forEach((category) => {
      const total = overview[category].total;
      overview[category].remaining = total - overview[category].totalUsed;
      if (total > 0) {
        Object.keys(overview[category].subcategories).forEach((subcategory) => {
          const amount = overview[category].subcategories[subcategory].amount;
          overview[category].subcategories[subcategory].percentage = (amount / total) * 100;
        });
      }
    });

    // Calculate overall totals
    const totalAvailable = Object.values(overview).reduce((sum, cat) => sum + cat.total, 0);
    const totalUsed = Object.values(overview).reduce((sum, cat) => sum + cat.totalUsed, 0);
    const remainingAmount = totalAvailable - totalUsed;

    res.status(200).json({
      success: true,
      overview,
      summary: {
        totalAvailable,
        totalUsed,
        remainingAmount
      }
    });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
});

export default router;