import { useQuery } from '@tanstack/react-query';
import { useEffect } from 'react';
import { get_conversation_history } from '../queries/conversations';
import useChatHistory from '../store/chat-history';
import { MESSAGE_TYPE, QUERYKEYS } from '../constant';

const useConversationController = () => {
  const {
    currentConversation,
    setCurrentConversation,
    resetChat,
    chatHistory,
    pushMessage,
    isStreaming,
    streaming,
    isThinking,
    toggleThinking,
    setChatHistory,
  } = useChatHistory();

  const { data, refetch: fetchChatHistory } = useQuery({
    queryKey: [QUERYKEYS.get_chat_history],
    queryFn: () => get_conversation_history(currentConversation),
    enabled: false,
  });

  const conversationHistory = data?.data ?? {};
  useEffect(() => {
    if (!data) return;
    const conversationHistory = data?.data?.items ?? [];
    setChatHistory(
      conversationHistory.map((item) => ({
        type: item.type === 'user' ? MESSAGE_TYPE.USER_MESSAGE : MESSAGE_TYPE.AGANT_RESPONSE,
        content: item.type === 'user' ? item.content : item.answer,
      })),
    );
  }, [data]);

  useEffect(() => {
    if (currentConversation !== null) {
      fetchChatHistory();
    } else {
      resetChat();
    }
  }, [currentConversation, fetchChatHistory]);

  return {
    currentConversation,
    setCurrentConversation,
    resetChat,
    chatHistory,
    pushMessage,
    isStreaming,
    streaming,
    isThinking,
    toggleThinking,
  };
};

export default useConversationController;
