import Box from '@mui/material/Box';
import { grey } from '@mui/material/colors';
import { IconButton, TextField, CircularProgress, Typography, Divider } from '@mui/material';
import DialogTitle from '@mui/material/DialogTitle';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import { useEffect, useRef, useState } from 'react';
import SendIcon from '@mui/icons-material/Send';
import { sendWS } from '../queries/wsClient';
import useChatHistory from '../store/chat-history';
import { MESSAGE_TYPE, TEXT } from '../constant/index';
import useConversationController from '../helpers/conversationController';

const response_style = {
  lineHeight: '22px',
  fontSize: '16px',
  marginBottom: '8px',
};

const initCitationDialog = { isOpen: false, title: '', content: '' };

const Chat = () => {
  const { currentConversation, chatHistory, pushMessage, isStreaming, streaming, isThinking, toggleThinking } = useConversationController();
  const chatAreaRef = useRef(null);
  const [citationDialog, setCitationDialog] = useState(initCitationDialog);
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

  const handleViewCitation = (citation) => {
    if (!citation) return;
    setCitationDialog({
      isOpen: true,
      title: `${citation.doc_title || 'Unknown title'} - Page ${citation.page || 'Unknown page'}`,
      content: citation.content || 'Unknown content',
    });
  };
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
          const citations = message?.citations;
          return (
            <Box key={index} sx={userMessage ? { display: 'flex', justifyContent: 'flex-end' } : {}}>
              <Typography
                sx={userMessage ? { maxWidth: '70%', padding: '12px', background: grey[100], borderRadius: '8px' } : response_style}
              >
                {message.content}
              </Typography>
              {!userMessage && citations && citations.length > 0 && (
                <Box id="message-citation">
                  {citations.map((citation, index) => (
                    <Box key={`${message.message_id}-${citation.chunk_id}-${index}`}>
                      <Divider />
                      <Box
                        sx={{ padding: '8px', cursor: 'pointer', borderRadius: '8px', '&:hover': { background: grey[200] } }}
                        onClick={() => handleViewCitation(citation)}
                      >
                        <Typography sx={{ fontStyle: 'italic', color: grey[500], fontSize: 14 }}>
                          {`Citation ${index + 1}: ${citation.doc_title}, page ${citation.page}`}
                        </Typography>
                        <Typography
                          sx={{
                            fontSize: '14px',
                            color: grey[900],
                            display: '-webkit-box',
                            WebkitBoxOrient: 'vertical',
                            WebkitLineClamp: 2,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                        >
                          {`${citation.content}`}
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </Box>
              )}
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
      <Box sx={{ display: chatHistory.length > 0 ? 'none' : 'block' }}>
        <Typography sx={{ textAlign: 'center', padding: 4 }} variant="h6">
          {TEXT.ask}
        </Typography>
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

      <Dialog onClose={() => setCitationDialog(initCitationDialog)} open={citationDialog.isOpen}>
        <DialogTitle>{citationDialog.title}</DialogTitle>
        <DialogContent>
          <Typography sx={{ whiteSpace: 'pre-line' }}>{citationDialog.content}</Typography>
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default Chat;
