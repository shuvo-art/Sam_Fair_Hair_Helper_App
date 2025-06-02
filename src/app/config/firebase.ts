import admin from 'firebase-admin';

// Load Firebase service account from environment variable
const serviceAccountRaw = process.env.FIREBASE_SERVICE_ACCOUNT;

let serviceAccount: admin.ServiceAccount;
if (serviceAccountRaw) {
  try {
    // Step 1: Parse the initial JSON string
    let parsedServiceAccount = JSON.parse(serviceAccountRaw);

    // Step 2: Handle the private_key to restore proper PEM format
    if (parsedServiceAccount.private_key) {
      // Replace \\n with actual newlines and format as multi-line PEM
      parsedServiceAccount.private_key = parsedServiceAccount.private_key
        .replace(/\\n/g, '\n') // Convert escaped newlines to actual newlines
        .trim(); // Remove trailing spaces
    }

    // Step 3: Assign to serviceAccount
    serviceAccount = parsedServiceAccount as admin.ServiceAccount;
  } catch (error) {
    throw new Error(`Failed to process FIREBASE_SERVICE_ACCOUNT: ${error instanceof Error ? error.message : String(error)}`);
  }
} else {
  throw new Error('FIREBASE_SERVICE_ACCOUNT environment variable is not set');
}

if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
  });
}

export const messaging = admin.messaging();