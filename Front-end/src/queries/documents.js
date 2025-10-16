import axios from 'axios';

const BASE_URL = 'http://0.0.0.0:8000';

export const upload_file = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  try {
    const response = await axios.post(`${BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response;
  } catch (err) {
    throw new Error(err);
  }
};

export const get_uploaded_documents = async () => {
  try {
    const response = await axios.get(`${BASE_URL}/documents`);
    return response;
  } catch (err) {
    throw new Error(err);
  }
};

export const delete_document = async (document_id) => {
  try {
    const response = await axios.delete(`${BASE_URL}/documents/${document_id}`);
    return response;
  } catch (err) {
    throw new Error(err);
  }
};
