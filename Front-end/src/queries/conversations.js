import axios from 'axios';

const BASE_URL = 'http://0.0.0.0:8000';

export const get_conversations = async () => {
  try {
    const response = await axios.get(`${BASE_URL}/conversations`);
    return response;
  } catch (err) {
    throw new Error(err);
  }
};

export const delete_conversation = async (conversation_id) => {
  try {
    const response = await axios.delete(`${BASE_URL}/conversations/${conversation_id}`);
    return response;
  } catch (err) {
    throw new Error(err);
  }
};

export const get_conversation_history = async (conversation_id) => {
  try {
    const response = await axios.get(`${BASE_URL}/conversations/${conversation_id}/history`);
    return response;
  } catch (err) {
    throw new Error(err);
  }
};
