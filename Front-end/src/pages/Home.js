import { useEffect } from 'react';
import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import Chat from '../components/Chat';
import SideMenu from '../components/SideMenu';
import { connectWS } from '../queries/wsClient';
import useChatHistory from '../store/chat-history';
import { MESSAGE_TYPE, NOTIFY_TYPE, QUERYKEYS } from '../constant/index';
import { useQueryClient } from '@tanstack/react-query';

const Home = () => {
  const { pushMessage, pushStreaming, toggleStreaming, clearStreaming, toggleThinking, setCurrentConversation } = useChatHistory();
  const queryClient = useQueryClient();
  useEffect(() => {
    connectWS((message) => {
      if (message.type === NOTIFY_TYPE.DELTA) {
        toggleThinking(false);
        toggleStreaming(true);
        pushStreaming({ type: MESSAGE_TYPE.STREAMING, content: message.text });
      }
      if (message.type === NOTIFY_TYPE.DONE) {
        toggleThinking(false);
        pushMessage({ type: MESSAGE_TYPE.AGANT_RESPONSE, content: message.answer });
        toggleStreaming(false);
        clearStreaming();
      }
      if (message.type === NOTIFY_TYPE.CONV_CREATE) {
        setCurrentConversation(message.conversation_id);
        queryClient.invalidateQueries({ queryKey: [QUERYKEYS.get_conversations] });
      }
    });
  }, []);
  return (
    <Box id="app-container" sx={{ width: '100%', height: '100%', display: 'flex' }}>
      <Box sx={{ width: 320, height: '100%' }}>
        <SideMenu></SideMenu>
      </Box>
      <Divider orientation="vertical" />
      <Box sx={{ flex: 1, display: 'flex', justifyContent: 'center', height: '100%' }}>
        <Chat></Chat>
      </Box>
    </Box>
  );
};

export default Home;
