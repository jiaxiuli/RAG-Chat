import Box from '@mui/material/Box';
import { grey } from '@mui/material/colors';
import { IconButton, TextField, CircularProgress, Typography } from '@mui/material';
import { useEffect, useRef, useState } from 'react';
import SendIcon from '@mui/icons-material/Send';
import { sendWS } from '../queries/wsClient';
import useChatHistory from '../store/chat-history';
import { MESSAGE_TYPE, TEXT } from '../constant/index';
import useConversationController from '../helpers/conversationController';

const response_style = {
  lineHeight: '22px',
  fontSize: '16px',
};

const Chat = () => {
  const { currentConversation, chatHistory, pushMessage, isStreaming, streaming, isThinking, toggleThinking } = useConversationController();
  const chatAreaRef = useRef(null);
  const [query, setQuery] = useState('');
  const handleSendMessage = () => {
    toggleThinking(true);
    pushMessage({ type: MESSAGE_TYPE.USER_MESSAGE, content: query });
    sendWS(query, currentConversation);
    setQuery('');
  };
  const handleKeyDownOnInput = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSendMessage();
    }
  };

  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  });
  return (
    <Box
      sx={{
        height: '100%',
        maxWidth: 800,
        width: '100%',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        boxSizing: 'border-box',
        gap: 2,
      }}
    >
      <Box
        className="chat-area"
        sx={{ minHeight: 0, flex: 1, display: 'flex', flexDirection: 'column', gap: 4, padding: '8px', overflow: 'auto' }}
        ref={chatAreaRef}
      >
        {chatHistory.map((message, index) => {
          const userMessage = message.type === MESSAGE_TYPE.USER_MESSAGE;
          return (
            <Box key={index} sx={userMessage ? { display: 'flex', justifyContent: 'flex-end' } : {}}>
              <Typography
                sx={userMessage ? { maxWidth: '70%', padding: '12px', background: grey[100], borderRadius: '8px' } : response_style}
              >
                {message.content}
              </Typography>
            </Box>
          );
        })}
        {isStreaming ? <Typography sx={response_style}>{streaming.map((msg) => msg.content).join(' ')}</Typography> : null}
        {isThinking && !isStreaming ? (
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <CircularProgress size={16} />
            <Typography sx={response_style}>{TEXT.thinking}</Typography>
          </Box>
        ) : null}
      </Box>
      <Box className="input-area" sx={{ height: chatHistory.length > 0 ? 'auto' : '50%' }}>
        <TextField
          multiline
          fullWidth
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDownOnInput}
          slotProps={{
            input: {
              sx: { borderRadius: '28px', padding: '8px 16px' },
              endAdornment: (
                <IconButton onClick={handleSendMessage}>
                  <SendIcon />
                </IconButton>
              ),
            },
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              '& fieldset': {
                boxShadow: 'rgba(149, 157, 165, 0.2) 0px 8px 24px',
              },
              '&:hover fieldset': {
                borderColor: 'rgba(0, 0, 0, 0.23)',
                borderWidth: 1,
              },
              '&.Mui-focused fieldset': {
                borderColor: 'rgba(0, 0, 0, 0.23)',
                borderWidth: 1,
              },
            },
          }}
        />
      </Box>
    </Box>
  );
};

export default Chat;
