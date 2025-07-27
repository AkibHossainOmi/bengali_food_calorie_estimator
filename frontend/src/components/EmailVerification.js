import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { authAPI } from '../utils/apiService';

export default function EmailVerification() {
  const location = useLocation();
  const [message, setMessage] = useState('Verifying your email...');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const token = params.get('token');

    if (!token) {
      setError('Missing verification token.');
      setLoading(false);
      return;
    }

    authAPI.verifyEmail(token)
      .then((res) => {
        // Show backend message as is, but optionally map for friendly UI
        if (res.msg === 'Account already verified') {
          setMessage('Your email is already verified. You can log in.');
        } else {
          setMessage('Email verified successfully! You can now log in.');
        }
        setLoading(false);
      })
      .catch(err => {
        setError(err.response?.data?.detail || err.message || 'Verification failed.');
        setLoading(false);
      });
  }, [location.search]);

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-100 px-4">
      {loading ? (
        <p className="text-indigo-600 text-xl font-semibold animate-pulse">Verifying your email...</p>
      ) : (
        <div className="max-w-md w-full p-8 bg-white shadow-md rounded-lg text-center">
          {error ? (
            <p className="text-red-600 font-semibold">{error}</p>
          ) : (
            <>
              <p className="text-green-600 font-semibold mb-6">{message}</p>
              <a
                href="/login"
                className="inline-block mt-4 px-6 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors"
              >
                Go to Login
              </a>
            </>
          )}
        </div>
      )}
    </div>
  );
}
