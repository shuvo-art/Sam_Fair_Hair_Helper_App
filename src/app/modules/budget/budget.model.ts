import mongoose, { Schema, Document } from 'mongoose';
import { IUser } from '../user/user.model';

// Interface for a single budget entry (category, subcategory, amount)
export interface IBudgetEntry {
  category: string;
  subcategory: string;
  amount: number;
}

// Interface for the budget document
export interface IBudget extends Document {
  userId: mongoose.Types.ObjectId;
  entries: IBudgetEntry[];
  startDate: Date;
  endDate: Date;
}

// Define allowed subcategories for each category
const categorySubcategoryMap: Record<string, string[]> = {
  'Core Supports': [
    'Assistance with daily life',
    'Assistance with social, economic, and community participation',
    'Consumables',
    'Transport'
  ],
  'Capacity Building Supports': [
    'Support coordination',
    'Improved living arrangements',
    'Increased social and community participation',
    'Finding and keeping a job',
    'Improved relationships',
    'Improved health and well-being',
    'Improved learning',
    'Improved life choices',
    'Improved daily living'
  ],
  'Capital Supports': [
    'Assistive technology',
    'Equipment',
    'Vehicle modifications',
    'Home modifications',
    'Specialist disability accommodation'
  ]
};

// Schema for a single budget entry
const BudgetEntrySchema: Schema = new Schema({
  category: { 
    type: String, 
    required: true, 
    enum: Object.keys(categorySubcategoryMap)
  },
  subcategory: { 
    type: String, 
    required: true,
    validate: {
      validator: function(this: any, value: string) {
        // Ensure subcategory belongs to the selected category
        return categorySubcategoryMap[this.category]?.includes(value);
      },
      message: (props: any) => `${props.value} is not a valid subcategory for category ${props.instance.category}`
    }
  },
  amount: { type: Number, required: true, min: 0 }
});

const BudgetSchema: Schema = new Schema(
  {
    userId: { type: mongoose.Types.ObjectId, ref: 'User', required: true },
    entries: [BudgetEntrySchema], // Array of budget entries
    startDate: { type: Date, required: true },
    endDate: { type: Date, required: true },
  },
  { timestamps: true }
);

// Ensure at least one entry exists
BudgetSchema.path('entries').validate((entries: IBudgetEntry[]) => {
  return entries && entries.length > 0;
}, 'At least one budget entry is required.');

export const Budget = mongoose.model<IBudget>('Budget', BudgetSchema);