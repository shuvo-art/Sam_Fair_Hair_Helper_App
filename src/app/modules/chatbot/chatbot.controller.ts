import { RequestHandler } from 'express';
import { ChatHistory } from './chatHistory.model';
import { Budget } from '../budget/budget.model';
import { exec, execSync } from 'child_process';
import path from 'path';
import fs from 'fs';
import { promisify } from 'util';
import { uploadImage } from '../../utils/cloudinary';

const execPromise = promisify(exec);

interface UserInputData {
  text_input?: string;
  pdf_path?: string;
  image_path?: string;
  budget_info?: string;
  conversation_history?: string;
}

interface ResponseData {
  response: string;
}

// Check if Python is available and get the executable
function getPythonExecutable(): string {
  const possibleExecutables = ['python', 'python3', 'py'];
  for (const exec of possibleExecutables) {
    try {
      execSync(`${exec} --version`, { stdio: 'ignore' });
      return exec;
    } catch (error) {
      continue;
    }
  }
  throw new Error('Python executable not found. Please ensure Python is installed and in the PATH.');
}

async function processAdminPdfUpload(pdf_path: string): Promise<ResponseData> {
  const pythonDir = path.join(__dirname, '../../../../samfair');
  const pythonScriptPath = path.join(pythonDir, 'Upload.py');
  const pythonExec = getPythonExecutable();
  const command = `${pythonExec} -u "${pythonScriptPath}" "${pdf_path}"`;

  console.log('Executing Python command for admin PDF upload:', command);
  console.log('Python directory:', pythonDir);
  console.log('Environment PATH:', process.env.PATH);

  try {
    const { stdout, stderr } = await execPromise(command, {
      cwd: pythonDir,
      // Remove shell: 'cmd.exe' to use default shell
    });
    console.log('Python script stdout:', stdout);
    console.log('Python script stderr:', stderr);

    const responseMatch = stdout.match(/AI Response: ([\s\S]+)/);
    const response = responseMatch ? responseMatch[1].trim() : 'No response generated';
    return { response };
  } catch (error: any) {
    console.error('Error executing Python script for admin PDF:', error.message);
    console.error('Command output:', error.stdout || 'No stdout');
    console.error('Command error output:', error.stderr || 'No stderr');
    throw new Error(`Command failed: ${error.message}${error.stderr ? `\n${error.stderr}` : ''}`);
  }
}

async function processUserChatInput({
  text_input,
  pdf_path,
  image_path,
  budget_info,
  conversation_history,
}: UserInputData): Promise<ResponseData> {
  const pythonDir = path.join(__dirname, '../../../../samfair');
  const pythonScriptPath = path.join(pythonDir, 'Chat.py');
  const pythonExec = getPythonExecutable();
  let command = `${pythonExec} -u "${pythonScriptPath}"`;

  if (text_input) command += ` "${text_input.replace(/"/g, '\\"')}"`;
  if (image_path) command += ` --image "${image_path}"`;
  if (pdf_path) command += ` --pdf "${pdf_path}"`;
  if (budget_info) {
    const escapedBudgetInfo = budget_info.replace(/"/g, '\\"');
    command += ` --budget """${escapedBudgetInfo}"""`;
  }
  if (conversation_history) {
    const escapedHistory = conversation_history.replace(/"/g, '\\"');
    command += ` --history """${escapedHistory}"""`;
  }

  console.log('Executing Python command for user chat input:', command);
  console.log('Python directory:', pythonDir);
  console.log('Environment PATH:', process.env.PATH);

  try {
    const { stdout, stderr } = await execPromise(command, {
      cwd: pythonDir,
      // Remove shell: 'cmd.exe' to use default shell
    });
    console.log('Python script stdout:', stdout);
    console.log('Python script stderr:', stderr);

    const responseMatch = stdout.match(/AI Response: ([\s\S]+)/);
    const response = responseMatch ? responseMatch[1].trim() : 'No response generated';
    return { response };
  } catch (error: any) {
    console.error('Error executing Python script for user chat:', error.message);
    console.error('Command output:', error.stdout || 'No stdout');
    console.error('Command error output:', error.stderr || 'No stderr');
    throw new Error(`Command failed: ${error.message}${error.stderr ? `\n${error.stderr}` : ''}`);
  }
}

// Admin-only PDF upload handler
export const handlePdfUpload: RequestHandler = async (req, res) => {
  try {
    const userId = req.user?.id;
    const file = req.file;

    console.log('Received userId:', userId);
    console.log('Received file:', file);

    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    if (!file) {
      res.status(400).json({ success: false, message: 'PDF file is required.' });
      return;
    }

    const pdfPath = path.resolve(file.path);
    console.log('PDF file path:', pdfPath);

    const response = await processAdminPdfUpload(pdfPath);

    if (fs.existsSync(pdfPath)) fs.unlinkSync(pdfPath);
    console.log('Temporary PDF file cleaned up');

    res.status(201).json({ success: true, message: response.response });
  } catch (error: any) {
    console.error('Error in handlePdfUpload:', error.message);
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ success: false, message: error.message });
  }
};

// User chat message handler
export const handleChatMessage: RequestHandler = async (req, res) => {
  try {
    const userId = req.user?.id;
    const { userMessage, chatId } = req.body;
    const file = req.file;

    console.log('Received userId:', userId);
    console.log('Received userMessage:', userMessage);
    console.log('Received chatId:', chatId);
    console.log('Received file:', file);

    if (!userId) {
      res.status(401).json({ success: false, message: 'Unauthorized' });
      return;
    }

    if (!userMessage && !file) {
      res.status(400).json({ success: false, message: 'Text, image, or PDF input is required.' });
      return;
    }

    const budgets = await Budget.find({ userId });
    const budgetInfo = budgets.flatMap(budget =>
      budget.entries.map(entry => ({
        category: entry.category,
        subcategory: entry.subcategory,
        amount: entry.amount,
        startDate: budget.startDate,
        endDate: budget.endDate,
      }))
    );

    const budgetInfoString = JSON.stringify(budgetInfo);

    let imagePath: string | undefined;
    let imageUrl: string | undefined;
    let pdfPath: string | undefined;

    if (file) {
      if (file.mimetype.startsWith('image/')) {
        imagePath = path.resolve(file.path);
        const uploadResult = await uploadImage(imagePath);
        imageUrl = uploadResult.secure_url;
        console.log('Image uploaded to Cloudinary:', imageUrl);
      } else if (file.mimetype === 'application/pdf') {
        pdfPath = path.resolve(file.path);
        console.log('PDF file path:', pdfPath);
      }
    }

    let chatHistory = chatId ? await ChatHistory.findById(chatId) : null;
    if (!chatHistory) {
      console.log('Starting new chat session');
      chatHistory = new ChatHistory({ userId, chat_contents: [] });
    }

    let conversationHistoryString = '';
    if (chatHistory.chat_contents.length > 0) {
      conversationHistoryString = chatHistory.chat_contents
        .map(content => {
          if (content.sent_by === 'User') {
            return `User: ${content.text_content || (content.image_url ? '[Image]' : '')}`;
          } else if (content.sent_by === 'Bot') {
            return `Assistant: ${content.text_content}`;
          }
          return '';
        })
        .filter(line => line)
        .join('\n');
    }

    const userMessageId = getNextMessageId(chatHistory.chat_contents);
    console.log('Generated userMessageId:', userMessageId);

    const inputData: UserInputData = {
      text_input: userMessage,
      image_path: imagePath,
      pdf_path: pdfPath,
      budget_info: budgetInfoString,
      conversation_history: conversationHistoryString || undefined,
    };
    const botResponse = await processUserChatInput(inputData);
    console.log('Bot response received:', botResponse.response);

    const botMessageId = userMessageId + 1;

    if (userMessage || imageUrl || pdfPath) {
      chatHistory.chat_contents.push({
        id: userMessageId,
        sent_by: 'User',
        text_content: userMessage || '',
        timestamp: new Date(),
        image_url: imageUrl,
      });
    }

    chatHistory.chat_contents.push({
      id: botMessageId,
      sent_by: 'Bot',
      text_content: botResponse.response,
      timestamp: new Date(),
    });

    await chatHistory.save();
    console.log('Chat history saved:', chatHistory._id);

    if (imagePath && fs.existsSync(imagePath)) {
      fs.unlinkSync(imagePath);
      console.log('Temporary image file cleaned up');
    }
    if (pdfPath && fs.existsSync(pdfPath)) {
      fs.unlinkSync(pdfPath);
      console.log('Temporary PDF file cleaned up');
    }

    res.status(201).json({ success: true, chatHistory });
  } catch (error: any) {
    console.error('Error in handleChatMessage:', error.message);
    if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ success: false, message: error.message });
  }
};

const getNextMessageId = (chatContents: any[]): number => {
  return chatContents.length > 0 ? chatContents[chatContents.length - 1].id + 1 : 1;
};

// Other controllers unchanged
export const getAllChats: RequestHandler = async (req, res) => {
  try {
    const userId = req.user?.id;
    console.log('Fetching all chats for userId:', userId);

    const chatHistories = await ChatHistory.find({ userId }).select('chat_name chat_contents');

    const formattedChatHistories = chatHistories.map(chat => {
      const lastMessage = chat.chat_contents.length > 0
        ? chat.chat_contents.reduce((latest, current) =>
            new Date(latest.timestamp) > new Date(current.timestamp) ? latest : current
          )
        : null;

      return {
        _id: chat._id,
        chat_name: chat.chat_name,
        timestamp: lastMessage ? lastMessage.timestamp : null
      };
    });

    res.status(200).json({ success: true, chatHistories: formattedChatHistories });
  } catch (error: any) {
    console.error('Error in getAllChats:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};

export const getChatHistory: RequestHandler = async (req, res) => {
  try {
    const chatId = req.params.chatId;
    console.log('Fetching chat history for chatId:', chatId);
    const chatHistory = await ChatHistory.findById(chatId).populate('userId', 'name');
    if (!chatHistory) {
      console.log('Chat not found');
      res.status(404).json({ success: false, message: 'Chat not found.' });
      return;
    }
    res.status(200).json({ success: true, chatHistory });
  } catch (error: any) {
    console.error('Error in getChatHistory:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};

export const updateChatName: RequestHandler = async (req, res) => {
  try {
    const { chatId } = req.params;
    const { newChatName } = req.body;
    console.log('Updating chat name for chatId:', chatId, 'to:', newChatName);

    if (!newChatName) {
      console.log('New chat name not provided');
      res.status(400).json({ success: false, message: 'New chat name is required.' });
      return;
    }

    const chatHistory = await ChatHistory.findByIdAndUpdate(
      chatId,
      { chat_name: newChatName },
      { new: true }
    );

    if (!chatHistory) {
      console.log('Chat not found');
      res.status(404).json({ success: false, message: 'Chat not found.' });
      return;
    }

    res.status(200).json({ success: true, chatHistory });
  } catch (error: any) {
    console.error('Error in updateChatName:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};

export const toggleBotResponseLikeStatus: RequestHandler = async (req, res) => {
  try {
    const { chatId, messageId } = req.params;
    console.log('Toggling like status for chatId:', chatId, 'messageId:', messageId);

    const chatHistory = await ChatHistory.findById(chatId);
    if (!chatHistory) {
      console.log('Chat not found');
      res.status(404).json({ success: false, message: 'Chat not found.' });
      return;
    }

    const message = chatHistory.chat_contents.find(
      (content) => content.id === parseInt(messageId) && content.sent_by === 'Bot'
    );

    if (!message) {
      console.log('Bot message not found');
      res.status(404).json({ success: false, message: 'Bot message not found.' });
      return;
    }

    await chatHistory.save();
    console.log('Like status updated');
    res.status(200).json({ success: true, message: 'Bot response like status updated.', chatHistory });
  } catch (error: any) {
    console.error('Error in toggleBotResponseLikeStatus:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};

export const deleteChat: RequestHandler = async (req, res) => {
  try {
    const { chatId } = req.params;
    console.log('Deleting chat with chatId:', chatId);

    const chatHistory = await ChatHistory.findByIdAndDelete(chatId);
    if (!chatHistory) {
      console.log('Chat not found');
      res.status(404).json({ success: false, message: 'Chat not found.' });
      return;
    }

    res.status(200).json({ success: true, message: 'Chat deleted successfully.' });
  } catch (error: any) {
    console.error('Error in deleteChat:', error.message);
    res.status(500).json({ success: false, message: error.message });
  }
};