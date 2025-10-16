import {
  Box,
  IconButton,
  Divider,
  Button,
  Typography,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import { useState } from 'react';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import EditNoteIcon from '@mui/icons-material/EditNote';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { grey } from '@mui/material/colors';
import { TEXT, QUERYKEYS, AccordionType } from '../constant/index';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { get_uploaded_documents, delete_document, upload_file } from '../queries/documents';
import { get_conversations, delete_conversation } from '../queries/conversations';
import { formatTime, formatExactTime } from '../helpers/helpers';
import toast from 'react-hot-toast';
import useChatHistory from '../store/chat-history';

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const highlightState = {
  background: grey[300],
  cursor: 'pointer',
};

const MenuItem = ({ item, handleDelete, type, onItemClick, isSelected }) => {
  return (
    <Box
      sx={{
        padding: '8px',
        borderRadius: '8px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        ...(isSelected ? highlightState : {}),
        '&:hover': {
          ...highlightState,
          '& .delete-btn': {
            opacity: 1,
            visibility: 'visible',
          },
        },
      }}
      onClick={() => {
        onItemClick && onItemClick(item.id);
      }}
    >
      <Box>
        <Typography color="textPrimary" sx={{ fontSize: 14 }}>
          {item.title}
        </Typography>
        <Typography color="textDisabled" sx={{ fontSize: 12 }}>
          {type === AccordionType.files ? `${TEXT.uploadedAt} ${formatTime(item.created_at)}` : ''}
          {type === AccordionType.conversations ? formatExactTime(item.updated_at) : ''}
        </Typography>
      </Box>

      <IconButton
        aria-label="delete"
        size="small"
        className="delete-btn"
        sx={{
          opacity: 0,
          visibility: 'hidden',
        }}
        onClick={() => handleDelete(item.id)}
      >
        <DeleteOutlineIcon fontSize="inherit" color="error" />
      </IconButton>
    </Box>
  );
};

const SideMenu = () => {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(AccordionType.files);

  const { currentConversation, setCurrentConversation, clearCurrentConversation } = useChatHistory();

  const {
    data: uploadedFiles,
    isLoading,
    isSuccess,
    isError,
  } = useQuery({ queryKey: [QUERYKEYS.uploaded_docs], queryFn: get_uploaded_documents });

  const {
    data: conversationsData,
    isLoading: isConversationLoading,
    isSuccess: isConversationSuccess,
  } = useQuery({ queryKey: [QUERYKEYS.get_conversations], queryFn: get_conversations });

  const { items: conversations } = conversationsData?.data ?? {};

  const uploadFileMutation = useMutation({
    mutationFn: upload_file,
    onSuccess: () => {
      toast.success(TEXT.fileUploadSuccess, { id: 'file-upload' });
      queryClient.invalidateQueries({ queryKey: [QUERYKEYS.uploaded_docs] });
    },
    onError: () => {
      toast.error(TEXT.fileUploadError, { id: 'file-upload' });
    },
  });

  const deleteDocMutation = useMutation({
    mutationFn: delete_document,
    onSuccess: (response) => {
      toast.success(TEXT.fileDeleteSuccess);
      queryClient.invalidateQueries({ queryKey: [QUERYKEYS.uploaded_docs] });
    },
    onError: (err) => {
      toast.error(TEXT.fileDeleteError);
    },
  });

  const deleteConvMutation = useMutation({
    mutationFn: delete_conversation,
    onSuccess: (response) => {
      toast.success(TEXT.chatDeleteSuccess);
      clearCurrentConversation();
      queryClient.invalidateQueries({ queryKey: [QUERYKEYS.get_conversations] });
    },
    onError: (err) => {
      toast.error(TEXT.chatDeleteError);
    },
  });

  const { items, has_more, next_cursor } = uploadedFiles?.data ?? {};
  const handleDeleteDocument = (document_id) => {
    deleteDocMutation.mutate(document_id);
  };
  const handleAccordinChange = (panel) => {
    setExpanded(panel);
  };

  const handleDeleteConversation = (conversation_id) => {
    deleteConvMutation.mutate(conversation_id);
  };

  const handleSwitchConversation = (conversation_id) => {
    setCurrentConversation(conversation_id);
  };

  const handleCreateNewChat = () => {
    if (currentConversation === null) return;
    clearCurrentConversation();
  };

  const handleUploadFile = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      toast.loading(TEXT.fileProcess, { id: 'file-upload' });
      uploadFileMutation.mutate(file);
    } else {
      toast.error(TEXT.fileUploadInvalid);
    }
  };

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', background: grey[100] }}>
      <Box sx={{ padding: '16px', boxSizing: 'border-box', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Button component="label" role={undefined} size="large" variant="contained" startIcon={<CloudUploadIcon />}>
          {TEXT.uploadFiles}
          <VisuallyHiddenInput type="file" onChange={handleUploadFile} />
        </Button>
        <Button
          component="label"
          role={undefined}
          size="large"
          variant="outlined"
          startIcon={<EditNoteIcon />}
          onClick={handleCreateNewChat}
        >
          {TEXT.newChat}
        </Button>
      </Box>
      <Divider></Divider>
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <Accordion
          expanded={expanded === AccordionType.files}
          onChange={() => handleAccordinChange(AccordionType.files)}
          sx={{
            flex: expanded === AccordionType.files ? 1 : 'unset',
            display: 'flex',
            flexDirection: 'column',
            height: expanded === AccordionType.files ? 'calc(100% - 128px)' : 'auto',
            '& .MuiCollapse-root': { flex: expanded === AccordionType.files ? 1 : 'unset', overflow: 'auto' },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls="uploaded-files-content" id="uploaded-files-header">
            <Typography component="span">{TEXT.uploadedFiles}</Typography>
          </AccordionSummary>
          <AccordionDetails className="ssss" sx={{ padding: '0px 8px 8px 8px' }}>
            {isLoading && (
              <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress />
              </Box>
            )}
            {isSuccess && items && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {items.map((item) => (
                  <MenuItem item={item} key={item.id} handleDelete={handleDeleteDocument} type={AccordionType.files} />
                ))}
              </Box>
            )}
          </AccordionDetails>
        </Accordion>
        <Accordion
          expanded={expanded === AccordionType.conversations}
          onChange={() => handleAccordinChange(AccordionType.conversations)}
          sx={{
            flex: expanded === AccordionType.conversations ? 1 : 'unset',
            display: 'flex',
            flexDirection: 'column',
            height: expanded === AccordionType.conversations ? 'calc(100% - 128px)' : 'auto',
            '& .MuiCollapse-root': { flex: expanded === AccordionType.conversations ? 1 : 'unset', overflow: 'auto' },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls="conversations-content" id="conversations-header">
            <Typography component="span">{TEXT.conversations}</Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ padding: '0px 8px 8px 8px' }}>
            {isConversationLoading && (
              <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress />
              </Box>
            )}
            {isConversationSuccess && conversations && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {conversations.map((item) => (
                  <MenuItem
                    item={item}
                    key={item.id}
                    isSelected={item.id === currentConversation}
                    handleDelete={handleDeleteConversation}
                    onItemClick={handleSwitchConversation}
                    type={AccordionType.conversations}
                  />
                ))}
              </Box>
            )}
          </AccordionDetails>
        </Accordion>
      </Box>
    </Box>
  );
};
export default SideMenu;
