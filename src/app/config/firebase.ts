import admin from 'firebase-admin';
import serviceAccount from './somos-ai-voice-cloning-firebase-adminsdk-fbsvc-0793bb60e0.json'; 

// Initialize Firebase Admin SDK
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount as admin.ServiceAccount),
  });
}

export const messaging = admin.messaging();