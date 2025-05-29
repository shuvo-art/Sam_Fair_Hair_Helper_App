import { Socket } from 'socket.io';
import { IUser } from "./app/modules/user/user.model";

declare global {
  namespace Express {
    interface Request {
      user?: IUser;
    }
  }

  namespace SocketIO {
    interface CustomSocket extends Socket {
      user?: IUser;
    }
  }
}
export {};


