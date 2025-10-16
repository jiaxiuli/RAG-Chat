import { create } from 'zustand';

const useChatHistory = create((set) => ({
  currentConversation: null,
  isThinking: false,
  isStreaming: false,
  chatHistory: [],
  streaming: [],
  setCurrentConversation: (conversation_id) =>
    set((state) => ({
      currentConversation: conversation_id,
    })),
  clearCurrentConversation: (conversation_id) =>
    set((state) => ({
      currentConversation: null,
    })),
  pushMessage: (message) =>
    set((state) => ({
      chatHistory: [...state.chatHistory, message],
    })),
  pushStreaming: (message) =>
    set((state) => ({
      streaming: [...state.streaming, message],
    })),
  toggleStreaming: (flag) =>
    set((state) => ({
      isStreaming: flag,
    })),
  toggleThinking: (flag) => set((state) => ({ isThinking: flag })),
  clearStreaming: () => set((state) => ({ streaming: [] })),
  resetChat: () => set((state) => ({ chatHistory: [] })),
  setChatHistory: (chatHistory) => set((state) => ({ chatHistory })),
}));

export default useChatHistory;
