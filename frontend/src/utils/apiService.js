import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

const apiService = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to automatically add Authorization header if token exists
apiService.interceptors.request.use(config => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ============ AUTH API CALLS ============

export const authAPI = {
  register: (data) => apiService.post('/api/auth/register', data).then(res => res.data),
  login: (data) => apiService.post('/api/auth/login', new URLSearchParams(data), {
    headers: {'Content-Type': 'application/x-www-form-urlencoded'}
  }).then(res => res.data),
  verifyEmail: (token) => apiService.get(`/api/auth/verify-email?token=${token}`).then(res => res.data),
  passwordResetRequest: (data) => apiService.post('/api/auth/password-reset-request', data).then(res => res.data),
  passwordResetConfirm: (data) => apiService.post('/api/auth/password-reset-confirm', data).then(res => res.data),
};

// ============ FOOD API CALLS ============

export const foodAPI = {
  predictFood: (file) => {
    if (!file) throw new Error("No file provided for prediction");

    const formData = new FormData();
    formData.append("file", file);

    return apiService.post('/api/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }).then(res => res.data);
  },
};

// ============ GENERIC API CALL HELPER ============

export const apiCall = async (method, endpoint, data = null, config = {}) => {
  try {
    const response = await apiService({
      method,
      url: endpoint,
      data,
      ...config,
    });
    return { success: true, data: response.data };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.message || error.message || 'An error occurred',
      status: error.response?.status,
    };
  }
};

export default apiService;
