import express, { Request, Response } from 'express';
import { Budget } from './budget.model';
import { authenticate } from '../auth/auth.middleware';
import { z } from 'zod';

const router = express.Router();

// Validation schemas remain unchanged (as provided earlier)
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

const createBudgetSchema = z.object({
  entries: z.array(budgetEntrySchema).min(1, 'At least one budget entry is required'),
  startDate: z.string().transform((val) => new Date(val)),
  endDate: z.string().transform((val) => new Date(val)),
});

const updateBudgetSchema = z.object({
  entries: z.array(budgetEntrySchema).optional(),
  startDate: z.string().transform((val) => new Date(val)).optional(),
  endDate: z.string().transform((val) => new Date(val)).optional(),
});

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

// Helper function to calculate period elapsed percentage
const calculatePeriodElapsed = (startDate: Date, endDate: Date, currentDate: Date = new Date()) => {
  const totalDurationMs = endDate.getTime() - startDate.getTime();
  const elapsedMs = currentDate.getTime() - startDate.getTime();
  return totalDurationMs > 0 ? Math.min(100, Math.max(0, (elapsedMs / totalDurationMs) * 100)) : 0;
};

// Helper function to calculate weeks or months in duration
const getDurationUnits = (startDate: Date, endDate: Date, unit: 'weeks' | 'months') => {
  const msPerDay = 1000 * 60 * 60 * 24;
  const msPerWeek = msPerDay * 7;
  const msPerMonth = msPerDay * 30.436875; // Average days per month

  const totalDurationMs = endDate.getTime() - startDate.getTime();
  return Math.ceil(unit === 'weeks' ? totalDurationMs / msPerWeek : totalDurationMs / msPerMonth);
};

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

    const budget = await Budget.findOne({
      userId,
      startDate: { $lte: endDate },
      endDate: { $gte: startDate }
    });

    if (budget) {
      entries.forEach((newEntry: any) => {
        const existingEntryIndex = budget.entries.findIndex(
          (entry: any) => entry.category === newEntry.category && entry.subcategory === newEntry.subcategory
        );
        if (existingEntryIndex !== -1) {
          budget.entries[existingEntryIndex].amount = newEntry.amount;
        } else {
          // Ensure newEntry is a plain object before pushing
          budget.entries.push({ ...newEntry });
        }
      });
      budget.startDate = startDate;
      budget.endDate = endDate;
      await budget.save();
      res.status(201).json({ success: true, budget });
    } else {
      const newBudget = new Budget({
        userId,
        entries: entries.map(entry => ({ ...entry })), // Ensure entries are plain objects
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

    if (parsedData.startDate) budget.startDate = parsedData.startDate;
    if (parsedData.endDate) budget.endDate = parsedData.endDate;

    if (parsedData.entries) {
      parsedData.entries.forEach((newEntry: any) => {
        const existingEntryIndex = budget.entries.findIndex(
          (entry: any) => entry.category === newEntry.category && entry.subcategory === newEntry.subcategory
        );
        if (existingEntryIndex !== -1) {
          budget.entries[existingEntryIndex].amount = newEntry.amount;
        } else {
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

    parsedData.entries.forEach((newEntry: any) => {
      const existingEntryIndex = budget.entries.findIndex(
        (entry: any) => entry.category === newEntry.category && entry.subcategory === newEntry.subcategory
      );
      if (existingEntryIndex !== -1) {
        const entry = budget.entries[existingEntryIndex];
        const newUsedAmount = (entry.usedAmount || 0) + newEntry.usedAmount;
        if (newUsedAmount > entry.amount) {
          throw new Error(`Used amount for ${newEntry.subcategory} exceeds allocated amount of ${entry.amount}`);
        }
        budget.entries[existingEntryIndex].usedAmount = newUsedAmount;
      } else {
        throw new Error(`Entry with category ${newEntry.category} and subcategory ${newEntry.subcategory} not found`);
      }
    });

    await budget.save();

    const totalAvailable = budget.entries.reduce((sum, entry) => sum + entry.amount, 0);
    const totalUsed = budget.entries.reduce((sum, entry) => sum + (entry.usedAmount || 0), 0);
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
    const currentDate = new Date(); // Current date for period elapsed calculation

    // Group budgets by category and calculate totals
    const categories = ['Core Supports', 'Capacity Building Supports', 'Capital Supports'];
    const overview = categories.reduce((acc, category) => {
      acc[category] = {
        total: 0,
        totalUsed: 0,
        remaining: 0,
        percentage: 0,
        subcategories: {},
      };
      return acc;
    }, {} as Record<string, { total: number; totalUsed: number; remaining: number; percentage: number; subcategories: Record<string, { amount: number; usedAmount: number; percentage: number; remaining: number }> }>);

    budgets.forEach((budget) => {
      const periodElapsed = calculatePeriodElapsed(budget.startDate, budget.endDate, currentDate);
      const viewType = req.query.view as 'Weekly' | 'Monthly' | 'Total Budge' | undefined || 'Total Budge';
      const durationUnits = getDurationUnits(budget.startDate, budget.endDate, viewType === 'Weekly' ? 'weeks' : 'months');

      budget.entries.forEach((entry: any) => {
        const category = entry.category;
        const subcategory = entry.subcategory;
        let amount = entry.amount;
        let usedAmount = entry.usedAmount || 0;

        // Adjust values based on view type
        if (viewType === 'Weekly' || viewType === 'Monthly') {
          amount = amount / durationUnits;
          usedAmount = usedAmount / durationUnits;
        }

        overview[category].total += amount;
        overview[category].totalUsed += usedAmount;
        overview[category].subcategories[subcategory] = {
          amount,
          usedAmount,
          percentage: amount > 0 ? (usedAmount / amount) * 100 : 0,
          remaining: amount - usedAmount
        };
      });
    });

    // Calculate category-level percentages and remaining
    Object.keys(overview).forEach((category) => {
      const total = overview[category].total;
      overview[category].remaining = total - overview[category].totalUsed;
      overview[category].percentage = total > 0 ? (overview[category].totalUsed / total) * 100 : 0; // Add category percentage
      if (total > 0) {
        Object.keys(overview[category].subcategories).forEach((subcategory) => {
          const sub = overview[category].subcategories[subcategory];
          sub.percentage = (sub.usedAmount / sub.amount) * 100 || 0;
        });
      }
    });

    // Calculate overall totals
    const totalAvailable = Object.values(overview).reduce((sum, cat) => sum + cat.total, 0);
    const totalUsed = Object.values(overview).reduce((sum, cat) => sum + cat.totalUsed, 0);
    const remainingAmount = totalAvailable - totalUsed;

    // Calculate overall percentage
    const overallPercentage = totalAvailable > 0 ? (totalUsed / totalAvailable) * 100 : 0;

    // Calculate period elapsed for the first budget (assuming one active budget)
    const periodElapsed = budgets.length > 0 ? calculatePeriodElapsed(budgets[0].startDate, budgets[0].endDate) : 0;

    res.status(200).json({
      success: true,
      overview,
      summary: {
        totalAvailable,
        totalUsed,
        remainingAmount,
        percentage: overallPercentage, // Add overall percentage
        periodElapsed
      },
      viewType: req.query.view
    });
  } catch (error: any) {
    res.status(500).json({ success: false, message: error.message });
  }
});

export default router;