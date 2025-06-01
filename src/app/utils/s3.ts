import { S3Client, PutObjectCommand, ListObjectsV2Command, DeleteObjectCommand, GetObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';

dotenv.config();

// Initialize S3 client with IAM user credentials
const s3Client = new S3Client({
  region: process.env.AWS_REGION as string,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID as string,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY as string,
  },
});

// Helper function to generate presigned URL
const generatePresignedUrl = async (key: string): Promise<string> => {
  const command = new GetObjectCommand({
    Bucket: process.env.AWS_S3_BUCKET as string,
    Key: key,
  });
  return getSignedUrl(s3Client, command, { expiresIn: 3600 }); // URL expires in 1 hour
};

// Upload image to S3
export const uploadImage = async (filePath: string): Promise<{ secure_url: string }> => {
  try {
    const fileContent = fs.readFileSync(filePath);
    const fileName = path.basename(filePath);
    const key = `user_images/${fileName}`;

    const command = new PutObjectCommand({
      Bucket: process.env.AWS_S3_BUCKET as string,
      Key: key,
      Body: fileContent,
      ContentType: 'image/jpeg', // Adjust based on file type
    });

    await s3Client.send(command);
    const secure_url = await generatePresignedUrl(key);
    return { secure_url };
  } catch (error) {
    throw new Error(`Image upload failed: ${error instanceof Error ? error.message : JSON.stringify(error)}`);
  }
};

// Upload audio to S3
export const uploadAudio = async (filePath: string): Promise<{ secure_url: string }> => {
  try {
    const fileContent = fs.readFileSync(filePath);
    const fileName = path.basename(filePath);
    const key = `user_audio/${fileName}`;

    const command = new PutObjectCommand({
      Bucket: process.env.AWS_S3_BUCKET as string,
      Key: key,
      Body: fileContent,
      ContentType: 'audio/mpeg', // Adjust based on audio type (e.g., mp3, wav)
    });

    await s3Client.send(command);
    const secure_url = await generatePresignedUrl(key);
    return { secure_url };
  } catch (error) {
    throw new Error(`Audio upload failed: ${error instanceof Error ? error.message : JSON.stringify(error)}`);
  }
};

// Upload hero image to S3
export const uploadHeroImage = async (filePath: string): Promise<{ secure_url: string }> => {
  try {
    const fileContent = fs.readFileSync(filePath);
    const fileName = path.basename(filePath);
    const key = `hero_images/${fileName}`;

    const command = new PutObjectCommand({
      Bucket: process.env.AWS_S3_BUCKET as string,
      Key: key,
      Body: fileContent,
      ContentType: 'image/jpeg', // Adjust based on file type
    });

    await s3Client.send(command);
    const secure_url = await generatePresignedUrl(key);
    return { secure_url };
  } catch (error) {
    throw new Error(`Image upload failed: ${error instanceof Error ? error.message : JSON.stringify(error)}`);
  }
};

// Upload PDF to S3
export const uploadPdf = async (filePath: string): Promise<{ secure_url: string }> => {
  try {
    const fileContent = fs.readFileSync(filePath);
    const fileName = path.basename(filePath);
    const key = `user_pdfs/${fileName}`;

    const command = new PutObjectCommand({
      Bucket: process.env.AWS_S3_BUCKET as string,
      Key: key,
      Body: fileContent,
      ContentType: 'application/pdf',
    });

    await s3Client.send(command);
    const secure_url = await generatePresignedUrl(key);
    return { secure_url };
  } catch (error) {
    throw new Error(`PDF upload failed: ${error instanceof Error ? error.message : JSON.stringify(error)}`);
  }
};

// Retrieve hero images from S3
export const getHeroImages = async (): Promise<{ public_id: string; secure_url: string }[]> => {
  try {
    const command = new ListObjectsV2Command({
      Bucket: process.env.AWS_S3_BUCKET as string,
      Prefix: 'hero_images/',
      MaxKeys: 100,
    });

    const result = await s3Client.send(command);
    if (!result.Contents) {
      return [];
    }

    const images = await Promise.all(
      result.Contents.map(async (item) => {
        if (!item.Key) return null;
        const secure_url = await generatePresignedUrl(item.Key);
        return {
          public_id: item.Key,
          secure_url,
        };
      })
    );

    return images.filter((item): item is { public_id: string; secure_url: string } => item !== null);
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AccessDenied') {
        throw new Error('Failed to retrieve hero images: Access denied to S3 bucket. Please ask the root user to verify your IAM permissions.');
      }
      throw new Error(`Failed to retrieve hero images: ${error.message}`);
    }
    throw new Error(`Failed to retrieve hero images: ${JSON.stringify(error)}`);
  }
};

// Delete hero image from S3
export const deleteHeroImage = async (publicId: string): Promise<void> => {
  try {
    const command = new DeleteObjectCommand({
      Bucket: process.env.AWS_S3_BUCKET as string,
      Key: publicId,
    });

    await s3Client.send(command);
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AccessDenied') {
        throw new Error('Failed to delete hero image: Access denied to S3 bucket. Please ask the root user to verify your IAM permissions.');
      }
      throw new Error(`Failed to delete hero image: ${error.message}`);
    }
    throw new Error(`Failed to delete hero image: ${JSON.stringify(error)}`);
  }
};