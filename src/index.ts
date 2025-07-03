import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import { createServer } from 'http';
import mongoose from 'mongoose';
import cors from 'cors';
import { initializeSocketIO } from './socketIO';
import appRoutes from './app/server';
import subscriptionWebhook from './app/modules/subscription/subscription.controller';
import { startCronJobs } from './app/cronJobs';

const app = express();
const server = createServer(app);
const PORT = process.env.PORT || 5001;

// Configure CORS with multiple allowed origins
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(',').map(origin => origin.trim())
  : ['*']; // Default to '*' if ALLOWED_ORIGINS is not set

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (e.g., curl or Postman) and check against allowed origins
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'], // Explicitly allow these HTTP methods
  allowedHeaders: ['Content-Type', 'Authorization'], // Allow these headers
  credentials: true, // Allow cookies and auth headers if needed
}));

// Use raw middleware only for the Stripe webhook
app.use('/api/v1/subscription/stripe', express.raw({ type: 'application/json' }), subscriptionWebhook);

// Apply JSON parsing globally for other routes
app.use(express.json());
app.use('/api/v1', appRoutes);

initializeSocketIO(server);

// Health check endpoints
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'ok' });
});

app.get('/health/db', async (req, res) => {
  try {
    // Check MongoDB connection state
    if (mongoose.connection.readyState === 1) {
      res.status(200).send('Database connection healthy');
    } else {
      console.log('Health check failed: MongoDB not connected');
      res.status(200).send('Application starting');
    }
  } catch (error) {
    console.error('Health check error:', error);
    res.status(200).send('Application starting');
  }
});

// MongoDB connection with retry logic
const mongoUri = process.env.MONGO_URI as string;

// Log the MongoDB URI (with password masked)
const maskedUri = mongoUri.replace(/(mongodb:\/\/[^:]+:)([^@]+)(@.+)/, '$1****$3');
console.log(`Connecting to MongoDB: ${maskedUri}`);

// Connection options with increased timeouts
const mongooseOptions = {
  //useNewUrlParser: true,
  //useUnifiedTopology: true,
  serverSelectionTimeoutMS: 60000, // Increase timeout to 60 seconds
  socketTimeoutMS: 60000,
  connectTimeoutMS: 60000,
  maxPoolSize: 10,
};

// Connect with retry
const connectWithRetry = () => {
  console.log('MongoDB connection attempt...');
  mongoose.connect(mongoUri, mongooseOptions)
    .then(() => {
      console.log('MongoDB connected successfully');
      startCronJobs(); // Start cron jobs after DB connection
    })
    .catch(err => {
      console.error('MongoDB connection error:', err);
      console.log('Retrying in 5 seconds...');
      setTimeout(connectWithRetry, 5000);
    });
};

// Handle connection events
mongoose.connection.on('error', err => {
  console.error('MongoDB connection error:', err);
});

mongoose.connection.on('disconnected', () => {
  console.log('MongoDB disconnected, attempting to reconnect...');
  setTimeout(connectWithRetry, 5000);
});

// Start the server first
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  connectWithRetry(); // Then connect to MongoDB
});

// Handle server errors
server.on('error', (error) => {
  console.error('Server error:', error);
});

// Add global error handlers
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('HTTP server closed');
    mongoose.connection.close(false)
      .then(() => {
        console.log('MongoDB connection closed');
        process.exit(0);
      })
      .catch((err) => {
        console.error('Error closing MongoDB connection:', err);
        process.exit(1);
      });
  });
  setTimeout(() => {
    console.error('Forced shutdown after timeout');
    process.exit(1);
  }, 30000);
});

process.on('SIGINT', async () => {
  await mongoose.connection.close();
  console.log('MongoDB connection closed due to app termination');
  process.exit(0);
});