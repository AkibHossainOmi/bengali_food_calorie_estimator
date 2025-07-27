import React, { useState } from 'react';
import { authAPI } from '../utils/apiService';

export default function PasswordResetRequest() {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage(null);
    try {
      await authAPI.passwordResetRequest({ email });
      setMessage('If your email is registered, you will receive a reset email shortly.');
    } catch {
      setMessage('If your email is registered, you will receive a reset email shortly.');
    }
  };

  return React.createElement(
    'form',
    { onSubmit: handleSubmit },
    React.createElement('input', {
      type: 'email',
      placeholder: 'Email',
      value: email,
      onChange: (e) => setEmail(e.target.value),
      required: true,
    }),
    React.createElement('button', { type: 'submit' }, 'Request Password Reset'),
    message && React.createElement('p', null, message)
  );
}
