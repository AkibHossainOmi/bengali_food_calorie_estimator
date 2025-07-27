import React, { useState, useEffect } from 'react';
import { authAPI } from '../utils/apiService';

export default function PasswordResetConfirm({ location }) {
  const [token, setToken] = useState(null);
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    setToken(params.get('token'));
  }, [location.search]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage(null);
    setError(null);
    if (password !== confirm) {
      setError('Passwords do not match');
      return;
    }
    try {
      await authAPI.passwordResetConfirm({ token, new_password: password });
      setMessage('Password reset successful! You can now log in.');
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    }
  };

  if (!token) {
    return React.createElement('p', { style: { color: 'red' } }, 'Missing reset token');
  }

  return React.createElement(
    'form',
    { onSubmit: handleSubmit },
    React.createElement('input', {
      type: 'password',
      placeholder: 'New Password',
      value: password,
      onChange: (e) => setPassword(e.target.value),
      required: true,
    }),
    React.createElement('input', {
      type: 'password',
      placeholder: 'Confirm Password',
      value: confirm,
      onChange: (e) => setConfirm(e.target.value),
      required: true,
    }),
    React.createElement('button', { type: 'submit' }, 'Reset Password'),
    message && React.createElement('p', { style: { color: 'green' } }, message),
    error && React.createElement('p', { style: { color: 'red' } }, error)
  );
}
